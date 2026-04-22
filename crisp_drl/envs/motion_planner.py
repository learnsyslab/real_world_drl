"""Simple EE-space motion planner that streams pose targets to the CRISP
cartesian impedance controller.

Quintic time-warp in position + Slerp in orientation, single segment.
Designed for robust real-hardware testing: conservative defaults, workspace-box
safety, Ctrl-C-safe freeze, dry-run mode.

See plan at ~/.claude/plans/jiggly-purring-moonbeam.md for design notes and
limitations.
"""

from __future__ import annotations

import logging
import signal
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import gymnasium as gym
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, RotationSpline, Slerp

from crisp_py.utils.geometry import Pose


WorkspaceBox = Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
OnTick = Callable[[int, Pose, Pose], bool]


@dataclass
class PlanAndExecuteResult:
    reached: bool
    aborted: bool
    reason: str
    elapsed: float
    final_pose_error_m: float
    final_orient_error_rad: float
    waypoints: Optional[list[Pose]] = None
    trajectory_duration: Optional[float] = None
    final_velocity: Optional[np.ndarray] = None
    final_acceleration: Optional[np.ndarray] = None


def _angle_between(a: Rotation, b: Rotation) -> float:
    return float((a * b.inv()).magnitude())


def _get_robot(env: gym.Env):
    robot = getattr(env.unwrapped, "robot", None)
    if robot is None:
        raise AttributeError(
            "plan_and_execute requires env.unwrapped.robot (crisp_py Robot); "
            f"got {type(env.unwrapped).__name__} without that attribute."
        )
    return robot


def _quintic_s(u: float) -> float:
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def _slerp_probe(t_sim: float, slerp_start_t: float, slerp_end_t: float) -> float:
    """Map sim-time to slerp probe with quintic ease so angular vel → 0 at endpoints."""
    window = slerp_end_t - slerp_start_t
    if window <= 0.0:
        return slerp_end_t
    u = (t_sim - slerp_start_t) / window
    if u <= 0.0:
        return slerp_start_t
    if u >= 1.0:
        return slerp_end_t
    return slerp_start_t + _quintic_s(u) * window


def _interpolate(
    start: Pose, goal: Pose, slerp: Slerp, u: float
) -> Pose:
    s = _quintic_s(u)
    p = start.position + (goal.position - start.position) * s
    r = slerp([s])[0]
    return Pose(p, r)


def _inside_box(pos: np.ndarray, box: WorkspaceBox) -> bool:
    return all(lo <= v <= hi for v, (lo, hi) in zip(pos, box))


def plan_and_execute(
    env: gym.Env,
    target_pose: Pose,
    *,
    duration: Optional[float] = None,
    max_linear_vel: float = 0.05,
    max_angular_vel: float = 0.3,
    stream_rate: Optional[float] = None,
    on_tick: Optional[OnTick] = None,
    timeout: float = 10.0,
    pos_tol: float = 1e-3,
    rot_tol: float = 0.02,
    workspace_box: Optional[WorkspaceBox] = None,
    dry_run: bool = False,
    hold_after: bool = True,
    logger: Optional[logging.Logger] = None,
) -> PlanAndExecuteResult:
    """Plan and execute a single EE-space quintic + Slerp trajectory.

    Args:
        env: gym env whose .unwrapped exposes a crisp_py Robot as .robot.
        target_pose: crisp_py Pose (position + scipy Rotation).
        duration: segment duration in s. If None, derived from max_linear_vel
            and the Euclidean distance (floor 250 ms).
        max_linear_vel: speed cap used to derive duration (m/s).
        max_angular_vel: angular speed cap; extends duration if orientation
            delta would exceed it.
        stream_rate: Hz to push set_target at. Default = robot publish_frequency
            (usually 50 Hz). Streaming faster than that is wasted because the
            robot's internal ROS timer drops intermediate targets.
        on_tick: callback(k, waypoint, measured) -> abort?. Use for
            retargeting / custom safety. Return True to stop streaming.
        timeout: max seconds for the whole call (stream + settle).
        pos_tol, rot_tol: settle tolerances (m, rad).
        workspace_box: ((xmin,xmax),(ymin,ymax),(zmin,zmax)) in base frame.
            Rejects targets outside the box and aborts if measured pose
            leaves it during streaming. None disables (logs a warning).
        dry_run: plan only, do not call set_target. Returns waypoints in result.
        hold_after: after stream, keep publishing target_pose until pos_tol
            and rot_tol are met or timeout fires. If False, returns at end of
            stream with whatever error remains.
        logger: optional logger.

    Returns:
        PlanAndExecuteResult.
    """
    log = logger or logging.getLogger(__name__)
    robot = _get_robot(env)

    if stream_rate is None:
        stream_rate = float(getattr(robot.config, "publish_frequency", 50.0))

    if workspace_box is None:
        log.warning(
            "plan_and_execute called without workspace_box — no spatial safety clamp active."
        )

    start_pose = robot.end_effector_pose
    goal_pose = target_pose.copy()

    if workspace_box is not None and not _inside_box(goal_pose.position, workspace_box):
        return PlanAndExecuteResult(
            reached=False,
            aborted=True,
            reason="safety_violation_target_outside_box",
            elapsed=0.0,
            final_pose_error_m=float(np.linalg.norm(goal_pose.position - start_pose.position)),
            final_orient_error_rad=_angle_between(start_pose.orientation, goal_pose.orientation),
        )

    dist = float(np.linalg.norm(goal_pose.position - start_pose.position))
    ang = _angle_between(start_pose.orientation, goal_pose.orientation)

    if duration is None:
        dur_lin = dist / max(max_linear_vel, 1e-6)
        dur_ang = ang / max(max_angular_vel, 1e-6)
        duration = max(dur_lin, dur_ang, 0.25)

    effective_lin_vel = dist / duration if duration > 0 else 0.0
    if effective_lin_vel > max_linear_vel * 1.2:
        return PlanAndExecuteResult(
            reached=False,
            aborted=True,
            reason=f"safety_violation_speed_{effective_lin_vel:.3f}mps_gt_cap",
            elapsed=0.0,
            final_pose_error_m=dist,
            final_orient_error_rad=ang,
        )

    N = max(2, int(duration * stream_rate) + 1)
    key_rots = Rotation.concatenate([start_pose.orientation, goal_pose.orientation])
    slerp = Slerp([0.0, 1.0], key_rots)

    waypoints: list[Pose] = [
        _interpolate(start_pose, goal_pose, slerp, k / (N - 1)) for k in range(N)
    ]

    if dry_run:
        log.info(
            "plan_and_execute dry_run: N=%d dur=%.3fs dist=%.4fm ang=%.3frad "
            "v_lin=%.4fm/s",
            N,
            duration,
            dist,
            ang,
            effective_lin_vel,
        )
        return PlanAndExecuteResult(
            reached=False,
            aborted=False,
            reason="dry_run",
            elapsed=0.0,
            final_pose_error_m=dist,
            final_orient_error_rad=ang,
            waypoints=waypoints,
        )

    def _freeze():
        try:
            robot.set_target(pose=robot.end_effector_pose)
        except Exception:
            pass

    previous_sigint = signal.getsignal(signal.SIGINT)

    def _on_sigint(signum, frame):
        _freeze()
        signal.signal(signal.SIGINT, previous_sigint)
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGINT, _on_sigint)
    except ValueError:
        pass

    t0 = time.monotonic()
    dt = 1.0 / stream_rate
    aborted = False
    abort_reason = ""

    try:
        for k, wp in enumerate(waypoints):
            measured = robot.end_effector_pose
            if workspace_box is not None and not _inside_box(measured.position, workspace_box):
                aborted = True
                abort_reason = "safety_violation_measured_outside_box"
                break
            if on_tick is not None and on_tick(k, wp, measured):
                aborted = True
                abort_reason = "callback_abort"
                break

            robot.set_target(pose=wp)

            next_deadline = t0 + (k + 1) * dt
            sleep_for = next_deadline - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)

            if time.monotonic() - t0 > timeout:
                aborted = True
                abort_reason = "timeout"
                break

        if aborted:
            _freeze()
            measured = robot.end_effector_pose
            return PlanAndExecuteResult(
                reached=False,
                aborted=True,
                reason=abort_reason,
                elapsed=time.monotonic() - t0,
                final_pose_error_m=float(
                    np.linalg.norm(measured.position - goal_pose.position)
                ),
                final_orient_error_rad=_angle_between(
                    measured.orientation, goal_pose.orientation
                ),
                waypoints=waypoints,
            )

        if hold_after:
            while time.monotonic() - t0 < timeout:
                measured = robot.end_effector_pose
                err_p = float(np.linalg.norm(measured.position - goal_pose.position))
                err_r = _angle_between(measured.orientation, goal_pose.orientation)
                if err_p < pos_tol and err_r < rot_tol:
                    return PlanAndExecuteResult(
                        reached=True,
                        aborted=False,
                        reason="reached",
                        elapsed=time.monotonic() - t0,
                        final_pose_error_m=err_p,
                        final_orient_error_rad=err_r,
                        waypoints=waypoints,
                    )
                robot.set_target(pose=goal_pose)
                time.sleep(dt)

            measured = robot.end_effector_pose
            return PlanAndExecuteResult(
                reached=False,
                aborted=False,
                reason="timeout",
                elapsed=time.monotonic() - t0,
                final_pose_error_m=float(
                    np.linalg.norm(measured.position - goal_pose.position)
                ),
                final_orient_error_rad=_angle_between(
                    measured.orientation, goal_pose.orientation
                ),
                waypoints=waypoints,
            )

        measured = robot.end_effector_pose
        return PlanAndExecuteResult(
            reached=False,
            aborted=False,
            reason="stream_end_no_hold",
            elapsed=time.monotonic() - t0,
            final_pose_error_m=float(
                np.linalg.norm(measured.position - goal_pose.position)
            ),
            final_orient_error_rad=_angle_between(
                measured.orientation, goal_pose.orientation
            ),
            waypoints=waypoints,
        )

    finally:
        try:
            signal.signal(signal.SIGINT, previous_sigint)
        except ValueError:
            pass


def plan_and_execute_position(
    env: gym.Env,
    target_position: np.ndarray,
    **kwargs,
) -> PlanAndExecuteResult:
    """Convenience: move to target_position keeping current orientation."""
    robot = _get_robot(env)
    goal = Pose(np.asarray(target_position, dtype=np.float64), robot.end_effector_pose.orientation)
    return plan_and_execute(env, goal, **kwargs)


def _install_sigint_freeze(robot):
    previous = signal.getsignal(signal.SIGINT)

    def _freeze():
        try:
            robot.set_target(pose=robot.end_effector_pose)
        except Exception:
            pass

    def _handler(signum, frame):
        _freeze()
        signal.signal(signal.SIGINT, previous)
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGINT, _handler)
    except ValueError:
        pass
    return previous, _freeze


def _restore_sigint(previous):
    try:
        signal.signal(signal.SIGINT, previous)
    except ValueError:
        pass


def plan_and_execute_ruckig(
    env: gym.Env,
    target_pose: Pose,
    *,
    max_linear_vel: float = 0.1,
    max_linear_acc: float = 0.5,
    max_linear_jerk: float = 5.0,
    max_angular_vel: float = 0.3,
    stream_rate: Optional[float] = None,
    initial_velocity: Optional[np.ndarray] = None,
    initial_acceleration: Optional[np.ndarray] = None,
    target_velocity: Optional[np.ndarray] = None,
    target_acceleration: Optional[np.ndarray] = None,
    get_target_fn: Optional[Callable[[], Pose]] = None,
    on_tick: Optional[OnTick] = None,
    timeout: float = 10.0,
    pos_tol: float = 0.0035,
    rot_tol: float = 0.05,
    workspace_box: Optional[WorkspaceBox] = None,
    dry_run: bool = False,
    hold_after: bool = True,
    logger: Optional[logging.Logger] = None,
) -> PlanAndExecuteResult:
    """Online Ruckig 3D Cartesian position OTG + Slerp orientation.

    Ruckig generates time-optimal jerk-limited trajectories. Unlike the quintic
    version:
    - explicit max_linear_{vel,acc,jerk} caps at the trajectory level,
    - supports non-zero initial velocity/acceleration (continuous chaining),
    - supports on-the-fly retargeting via get_target_fn() each tick.

    Orientation is Slerp'd in parallel; if angular distance / T > max_angular_vel
    the Slerp end-time is extended so orientation finishes after position hold.
    """
    import ruckig  # local import so envs without ruckig still load the module

    log = logger or logging.getLogger(__name__)
    robot = _get_robot(env)

    if stream_rate is None:
        stream_rate = float(getattr(robot.config, "publish_frequency", 50.0))
    dt = 1.0 / stream_rate

    if workspace_box is None:
        log.warning(
            "plan_and_execute_ruckig called without workspace_box — no spatial safety clamp active."
        )

    start_pose = robot.end_effector_pose
    goal_pose = target_pose.copy()

    if workspace_box is not None and not _inside_box(goal_pose.position, workspace_box):
        return PlanAndExecuteResult(
            reached=False,
            aborted=True,
            reason="safety_violation_target_outside_box",
            elapsed=0.0,
            final_pose_error_m=float(
                np.linalg.norm(goal_pose.position - start_pose.position)
            ),
            final_orient_error_rad=_angle_between(
                start_pose.orientation, goal_pose.orientation
            ),
        )

    otg = ruckig.Ruckig(3, dt)
    inp = ruckig.InputParameter(3)
    out = ruckig.OutputParameter(3)

    v0 = (
        np.asarray(initial_velocity, dtype=np.float64)
        if initial_velocity is not None
        else np.zeros(3)
    )
    a0 = (
        np.asarray(initial_acceleration, dtype=np.float64)
        if initial_acceleration is not None
        else np.zeros(3)
    )

    vT = (
        np.asarray(target_velocity, dtype=np.float64)
        if target_velocity is not None
        else np.zeros(3)
    )
    aT = (
        np.asarray(target_acceleration, dtype=np.float64)
        if target_acceleration is not None
        else np.zeros(3)
    )

    inp.current_position = list(start_pose.position)
    inp.current_velocity = v0.tolist()
    inp.current_acceleration = a0.tolist()
    inp.target_position = list(goal_pose.position)
    inp.target_velocity = vT.tolist()
    inp.target_acceleration = aT.tolist()
    inp.max_velocity = [max_linear_vel] * 3
    inp.max_acceleration = [max_linear_acc] * 3
    inp.max_jerk = [max_linear_jerk] * 3

    res = otg.update(inp, out)
    if res == ruckig.Result.Error or int(res) < 0:
        return PlanAndExecuteResult(
            reached=False,
            aborted=True,
            reason=f"ruckig_init_error_{res}",
            elapsed=0.0,
            final_pose_error_m=float(
                np.linalg.norm(goal_pose.position - start_pose.position)
            ),
            final_orient_error_rad=_angle_between(
                start_pose.orientation, goal_pose.orientation
            ),
        )

    T = float(out.trajectory.duration)
    ang = _angle_between(start_pose.orientation, goal_pose.orientation)
    T_rot = ang / max(max_angular_vel, 1e-6)
    T_total = max(T, T_rot)

    slerp_start_t = 0.0
    slerp_end_t = T_total
    slerp = Slerp([slerp_start_t, slerp_end_t], Rotation.concatenate([start_pose.orientation, goal_pose.orientation]))

    if dry_run:
        log.info(
            "plan_and_execute_ruckig dry_run: T=%.3fs T_rot=%.3fs dist=%.4fm ang=%.3frad",
            T,
            T_rot,
            float(np.linalg.norm(goal_pose.position - start_pose.position)),
            ang,
        )
        # Build full waypoint list (position-only; orientation from slerp at same t).
        waypoints: list[Pose] = [
            Pose(np.asarray(out.new_position), slerp([_slerp_probe(0.0, slerp_start_t, slerp_end_t)])[0])
        ]
        t_sim = 0.0
        while res != ruckig.Result.Finished:
            inp.current_position = out.new_position
            inp.current_velocity = out.new_velocity
            inp.current_acceleration = out.new_acceleration
            res = otg.update(inp, out)
            t_sim += dt
            probe = _slerp_probe(t_sim, slerp_start_t, slerp_end_t)
            waypoints.append(Pose(np.asarray(out.new_position), slerp([probe])[0]))
            if t_sim > timeout:
                break
        return PlanAndExecuteResult(
            reached=False,
            aborted=False,
            reason="dry_run",
            elapsed=0.0,
            final_pose_error_m=float(
                np.linalg.norm(
                    np.asarray(out.new_position) - goal_pose.position
                )
            ),
            final_orient_error_rad=_angle_between(
                waypoints[-1].orientation, goal_pose.orientation
            ),
            waypoints=waypoints,
            trajectory_duration=T_total,
        )

    previous_sigint, _freeze = _install_sigint_freeze(robot)
    t0 = time.monotonic()
    # Sim-time tracker for slerp probe (advances by dt per Ruckig step).
    # Decoupled from wall-clock to avoid loop-jitter stutter. The pre-loop
    # otg.update() already advanced out.new_* to t=dt.
    t_sim = dt
    aborted = False
    abort_reason = ""
    k = 0

    try:
        while res != ruckig.Result.Finished:
            t_elapsed = time.monotonic() - t0
            if t_elapsed > timeout:
                aborted = True
                abort_reason = "timeout"
                break

            measured = robot.end_effector_pose
            if workspace_box is not None and not _inside_box(
                measured.position, workspace_box
            ):
                aborted = True
                abort_reason = "safety_violation_measured_outside_box"
                break

            # Optional online retarget.
            if get_target_fn is not None:
                try:
                    new_goal = get_target_fn()
                except Exception as e:  # noqa: BLE001
                    log.warning("get_target_fn raised %s; continuing with previous goal.", e)
                    new_goal = None
                if new_goal is not None and not np.allclose(
                    new_goal.position, goal_pose.position, atol=1e-6
                ):
                    if workspace_box is not None and not _inside_box(
                        new_goal.position, workspace_box
                    ):
                        aborted = True
                        abort_reason = "retarget_outside_box"
                        break
                    goal_pose = new_goal.copy()
                    inp.target_position = list(goal_pose.position)
                    # Re-key slerp from the current probed rotation to new goal
                    # over a fresh (t_sim, t_sim + T_rot_new) window. Using
                    # sim-time, not wall-clock, keeps orientation in lockstep
                    # with Ruckig position samples.
                    cur_probe = _slerp_probe(t_sim, slerp_start_t, slerp_end_t)
                    cur_rot = slerp([cur_probe])[0]
                    ang_remain = _angle_between(cur_rot, goal_pose.orientation)
                    T_rot_new = ang_remain / max(max_angular_vel, 1e-6)
                    slerp_start_t = t_sim
                    slerp_end_t = t_sim + max(T_rot_new, dt)
                    slerp = Slerp(
                        [slerp_start_t, slerp_end_t],
                        Rotation.concatenate([cur_rot, goal_pose.orientation]),
                    )

            # Step Ruckig for next sample.
            inp.current_position = out.new_position
            inp.current_velocity = out.new_velocity
            inp.current_acceleration = out.new_acceleration
            res = otg.update(inp, out)
            t_sim += dt

            probe = _slerp_probe(t_sim, slerp_start_t, slerp_end_t)
            rot = slerp([probe])[0]
            wp = Pose(np.asarray(out.new_position, dtype=np.float64), rot)

            if on_tick is not None and on_tick(k, wp, measured):
                aborted = True
                abort_reason = "callback_abort"
                break

            robot.set_target(pose=wp)
            k += 1

            next_deadline = t0 + (k + 1) * dt
            sleep_for = next_deadline - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)

        elapsed = time.monotonic() - t0
        final_v = np.asarray(out.new_velocity, dtype=np.float64).copy()
        final_a = np.asarray(out.new_acceleration, dtype=np.float64).copy()

        if aborted:
            _freeze()
            measured = robot.end_effector_pose
            return PlanAndExecuteResult(
                reached=False,
                aborted=True,
                reason=abort_reason,
                elapsed=elapsed,
                final_pose_error_m=float(
                    np.linalg.norm(measured.position - goal_pose.position)
                ),
                final_orient_error_rad=_angle_between(
                    measured.orientation, goal_pose.orientation
                ),
                trajectory_duration=T_total,
                final_velocity=final_v,
                final_acceleration=final_a,
            )

        if hold_after:
            while time.monotonic() - t0 < timeout:
                measured = robot.end_effector_pose
                err_p = float(np.linalg.norm(measured.position - goal_pose.position))
                err_r = _angle_between(measured.orientation, goal_pose.orientation)
                if err_p < pos_tol and err_r < rot_tol:
                    return PlanAndExecuteResult(
                        reached=True,
                        aborted=False,
                        reason="reached",
                        elapsed=time.monotonic() - t0,
                        final_pose_error_m=err_p,
                        final_orient_error_rad=err_r,
                        trajectory_duration=T_total,
                        final_velocity=final_v,
                        final_acceleration=final_a,
                    )
                robot.set_target(pose=goal_pose)
                time.sleep(dt)

            measured = robot.end_effector_pose
            return PlanAndExecuteResult(
                reached=False,
                aborted=False,
                reason="timeout",
                elapsed=time.monotonic() - t0,
                final_pose_error_m=float(
                    np.linalg.norm(measured.position - goal_pose.position)
                ),
                final_orient_error_rad=_angle_between(
                    measured.orientation, goal_pose.orientation
                ),
                trajectory_duration=T_total,
                final_velocity=final_v,
                final_acceleration=final_a,
            )

        measured = robot.end_effector_pose
        return PlanAndExecuteResult(
            reached=False,
            aborted=False,
            reason="stream_end_no_hold",
            elapsed=elapsed,
            final_pose_error_m=float(
                np.linalg.norm(measured.position - goal_pose.position)
            ),
            final_orient_error_rad=_angle_between(
                measured.orientation, goal_pose.orientation
            ),
            trajectory_duration=T_total,
            final_velocity=final_v,
            final_acceleration=final_a,
        )

    finally:
        _restore_sigint(previous_sigint)


def plan_and_execute_spline(
    env: gym.Env,
    waypoints: list[Pose],
    *,
    max_linear_vel: float = 0.1,
    max_linear_acc: float = 0.5,
    max_linear_jerk: float = 5.0,
    max_angular_vel: float = 0.3,
    stream_rate: Optional[float] = None,
    on_tick: Optional[OnTick] = None,
    timeout: float = 30.0,
    pos_tol: float = 0.0035,
    rot_tol: float = 0.05,
    workspace_box: Optional[WorkspaceBox] = None,
    dry_run: bool = False,
    hold_after: bool = True,
    num_arclen_samples: int = 512,
    logger: Optional[logging.Logger] = None,
) -> PlanAndExecuteResult:
    """Global C^2 spline through multiple Cartesian waypoints, Ruckig-retimed.

    Fits a clamped CubicSpline through [start, wp_1, ..., wp_N] in position and
    scipy RotationSpline through the same knots in orientation. Reparameterises
    by arc-length, then drives arc-length with a 1-DOF Ruckig OTG so the trans
    velocity / accel / jerk caps apply to the tangential motion.

    Unlike plan_and_execute_ruckig (single segment, stops at the goal), this
    function does NOT stop at intermediate waypoints — the robot glides through
    them along the spline. v=0 at the first and last knot only.

    Args:
        waypoints: list of crisp_py Pose, final entry is the goal. Start pose
            is read from robot.end_effector_pose and prepended internally.
        max_linear_vel/acc/jerk: Cartesian tangential envelope.
        max_angular_vel: if the orientation path would exceed this, the Ruckig
            arc-length caps are scaled down so angular motion fits.
        num_arclen_samples: density for numeric arc-length integration of the
            position spline (higher = more accurate arc-length param).
    """
    import ruckig

    log = logger or logging.getLogger(__name__)
    robot = _get_robot(env)

    if not waypoints:
        raise ValueError("plan_and_execute_spline requires at least one waypoint")

    if stream_rate is None:
        stream_rate = float(getattr(robot.config, "publish_frequency", 50.0))
    dt = 1.0 / stream_rate

    if workspace_box is None:
        log.warning(
            "plan_and_execute_spline called without workspace_box — no spatial safety clamp active."
        )

    start_pose = robot.end_effector_pose
    knot_poses: list[Pose] = [start_pose] + [wp.copy() for wp in waypoints]
    goal_pose = knot_poses[-1]

    if workspace_box is not None:
        for i, wp in enumerate(knot_poses):
            if not _inside_box(wp.position, workspace_box):
                return PlanAndExecuteResult(
                    reached=False,
                    aborted=True,
                    reason=f"safety_violation_waypoint_{i}_outside_box",
                    elapsed=0.0,
                    final_pose_error_m=float(
                        np.linalg.norm(goal_pose.position - start_pose.position)
                    ),
                    final_orient_error_rad=_angle_between(
                        start_pose.orientation, goal_pose.orientation
                    ),
                )

    positions = np.asarray([p.position for p in knot_poses], dtype=np.float64)
    rotations = Rotation.concatenate([p.orientation for p in knot_poses])

    # Chord-length knot parameter for both splines.
    seg_lens = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    if np.any(seg_lens < 1e-9):
        # Collapse duplicate knots so CubicSpline has strictly increasing x.
        keep = np.concatenate([[True], seg_lens > 1e-9])
        positions = positions[keep]
        rotations = Rotation.concatenate([rotations[i] for i in np.where(keep)[0]])
        seg_lens = np.linalg.norm(np.diff(positions, axis=0), axis=1)

    if len(positions) < 2 or float(np.sum(seg_lens)) < 1e-9:
        return PlanAndExecuteResult(
            reached=True,
            aborted=False,
            reason="noop_zero_distance",
            elapsed=0.0,
            final_pose_error_m=0.0,
            final_orient_error_rad=_angle_between(
                start_pose.orientation, goal_pose.orientation
            ),
            trajectory_duration=0.0,
        )

    us = np.concatenate([[0.0], np.cumsum(seg_lens)])
    u_max = float(us[-1])

    # Clamped BC: zero derivative (wrt spline param u) at both ends. Ruckig's
    # 1-DOF retime also starts/ends at v=0, so pose-velocity = ds/dt * dr/du = 0.
    if len(positions) >= 2:
        pos_spline = CubicSpline(us, positions, axis=0, bc_type="clamped")
    rot_spline = RotationSpline(us, rotations)

    # Arc-length LUT by dense sampling.
    u_dense = np.linspace(0.0, u_max, num_arclen_samples)
    p_dense = pos_spline(u_dense)
    d = np.linalg.norm(np.diff(p_dense, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(d)])
    L = float(arc[-1])

    def u_of_arc(s: float) -> float:
        return float(np.interp(s, arc, u_dense))

    # Angular path length (sum of consecutive inter-knot angles, upper bound on
    # actual RotationSpline arc since spline can overshoot slightly; good enough).
    ang_path = 0.0
    for i in range(len(rotations) - 1):
        ang_path += _angle_between(rotations[i], rotations[i + 1])

    # Ruckig 1-DOF on arc length s in [0, L].
    otg = ruckig.Ruckig(1, dt)
    inp = ruckig.InputParameter(1)
    out = ruckig.OutputParameter(1)
    inp.current_position = [0.0]
    inp.current_velocity = [0.0]
    inp.current_acceleration = [0.0]
    inp.target_position = [L]
    inp.target_velocity = [0.0]
    inp.target_acceleration = [0.0]
    inp.max_velocity = [max_linear_vel]
    inp.max_acceleration = [max_linear_acc]
    inp.max_jerk = [max_linear_jerk]

    res = otg.update(inp, out)
    if res == ruckig.Result.Error or int(res) < 0:
        return PlanAndExecuteResult(
            reached=False,
            aborted=True,
            reason=f"ruckig_init_error_{res}",
            elapsed=0.0,
            final_pose_error_m=float(
                np.linalg.norm(goal_pose.position - start_pose.position)
            ),
            final_orient_error_rad=ang_path,
        )

    T_lin = float(out.trajectory.duration)
    T_rot = ang_path / max(max_angular_vel, 1e-6)

    # If orientation needs more time than position, scale envelope down so the
    # 1D Ruckig takes T_rot. Preserve acc/jerk shape by scaling with the ratio.
    if T_rot > T_lin > 0.0:
        scale = T_lin / T_rot
        inp.max_velocity = [max_linear_vel * scale]
        inp.max_acceleration = [max_linear_acc * scale]
        inp.max_jerk = [max_linear_jerk * scale]
        res = otg.update(inp, out)

    T_total = max(T_lin, T_rot)

    def sample(s: float) -> Pose:
        s_c = max(0.0, min(s, L))
        u = u_of_arc(s_c)
        pos = np.asarray(pos_spline(u), dtype=np.float64)
        rot = rot_spline(u)
        return Pose(pos, rot)

    if dry_run:
        out_waypoints: list[Pose] = [sample(0.0)]
        t_sim = 0.0
        while res != ruckig.Result.Finished:
            inp.current_position = out.new_position
            inp.current_velocity = out.new_velocity
            inp.current_acceleration = out.new_acceleration
            res = otg.update(inp, out)
            t_sim += dt
            out_waypoints.append(sample(out.new_position[0]))
            if t_sim > timeout:
                break
        log.info(
            "plan_and_execute_spline dry_run: knots=%d L=%.4fm ang_path=%.3frad "
            "T_lin=%.3fs T_rot=%.3fs T_total=%.3fs",
            len(knot_poses),
            L,
            ang_path,
            T_lin,
            T_rot,
            T_total,
        )
        return PlanAndExecuteResult(
            reached=False,
            aborted=False,
            reason="dry_run",
            elapsed=0.0,
            final_pose_error_m=float(
                np.linalg.norm(out_waypoints[-1].position - goal_pose.position)
            ),
            final_orient_error_rad=_angle_between(
                out_waypoints[-1].orientation, goal_pose.orientation
            ),
            waypoints=out_waypoints,
            trajectory_duration=T_total,
        )

    previous_sigint, _freeze = _install_sigint_freeze(robot)
    t0 = time.monotonic()
    aborted = False
    abort_reason = ""
    k = 0

    try:
        while res != ruckig.Result.Finished:
            if time.monotonic() - t0 > timeout:
                aborted = True
                abort_reason = "timeout"
                break

            measured = robot.end_effector_pose
            if workspace_box is not None and not _inside_box(
                measured.position, workspace_box
            ):
                aborted = True
                abort_reason = "safety_violation_measured_outside_box"
                break

            inp.current_position = out.new_position
            inp.current_velocity = out.new_velocity
            inp.current_acceleration = out.new_acceleration
            res = otg.update(inp, out)

            wp = sample(out.new_position[0])

            if on_tick is not None and on_tick(k, wp, measured):
                aborted = True
                abort_reason = "callback_abort"
                break

            robot.set_target(pose=wp)
            k += 1

            next_deadline = t0 + (k + 1) * dt
            sleep_for = next_deadline - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)

        elapsed = time.monotonic() - t0

        if aborted:
            _freeze()
            measured = robot.end_effector_pose
            return PlanAndExecuteResult(
                reached=False,
                aborted=True,
                reason=abort_reason,
                elapsed=elapsed,
                final_pose_error_m=float(
                    np.linalg.norm(measured.position - goal_pose.position)
                ),
                final_orient_error_rad=_angle_between(
                    measured.orientation, goal_pose.orientation
                ),
                trajectory_duration=T_total,
            )

        if hold_after:
            while time.monotonic() - t0 < timeout:
                measured = robot.end_effector_pose
                err_p = float(np.linalg.norm(measured.position - goal_pose.position))
                err_r = _angle_between(measured.orientation, goal_pose.orientation)
                if err_p < pos_tol and err_r < rot_tol:
                    return PlanAndExecuteResult(
                        reached=True,
                        aborted=False,
                        reason="reached",
                        elapsed=time.monotonic() - t0,
                        final_pose_error_m=err_p,
                        final_orient_error_rad=err_r,
                        trajectory_duration=T_total,
                    )
                robot.set_target(pose=goal_pose)
                time.sleep(dt)

            measured = robot.end_effector_pose
            return PlanAndExecuteResult(
                reached=False,
                aborted=False,
                reason="timeout",
                elapsed=time.monotonic() - t0,
                final_pose_error_m=float(
                    np.linalg.norm(measured.position - goal_pose.position)
                ),
                final_orient_error_rad=_angle_between(
                    measured.orientation, goal_pose.orientation
                ),
                trajectory_duration=T_total,
            )

        measured = robot.end_effector_pose
        return PlanAndExecuteResult(
            reached=False,
            aborted=False,
            reason="stream_end_no_hold",
            elapsed=elapsed,
            final_pose_error_m=float(
                np.linalg.norm(measured.position - goal_pose.position)
            ),
            final_orient_error_rad=_angle_between(
                measured.orientation, goal_pose.orientation
            ),
            trajectory_duration=T_total,
        )

    finally:
        _restore_sigint(previous_sigint)

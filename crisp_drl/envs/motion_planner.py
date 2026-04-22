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
from scipy.spatial.transform import Rotation, Slerp

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

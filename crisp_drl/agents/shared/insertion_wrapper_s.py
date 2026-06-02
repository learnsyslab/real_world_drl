import threading
import time
from typing import Any, Dict, Optional
from gymnasium import Wrapper, spaces
import numpy as np

from crisp_drl.agents.shared.insertion_env_config import SiemensConfig
from crisp_drl.envs.pose_estimation_helper import (
    PoseEstimationHelper,
    euler_to_rot_matrix,
)
from crisp_drl.agents.shared.algorithm_config import Config

import contextlib

# Module-level abort flag. Set by the eval runner's keyboard listener when the
# operator presses 'u'. Checked at every _logged_env_step so the abort fires
# at the next control cycle regardless of where in the pipeline the robot is.
_abort_episode_flag: threading.Event = threading.Event()


class AbortEpisodeException(Exception):
    """Raised when the operator presses 'u' to abort the current episode."""


@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)


def rot_matrix_to_euler_xyz(rot: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to XYZ roll-pitch-yaw Euler angles."""
    sy = np.sqrt(rot[0, 0] ** 2 + rot[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(rot[2, 1], rot[2, 2])
        pitch = np.arctan2(-rot[2, 0], sy)
        yaw = np.arctan2(rot[1, 0], rot[0, 0])
    else:
        roll = np.arctan2(-rot[1, 2], rot[1, 1])
        pitch = np.arctan2(-rot[2, 0], sy)
        yaw = 0.0

    return np.array([roll, pitch, yaw])


class InsertionWrapperSiemens(Wrapper):
    def __init__(
        self,
        env,
        alg_config: Config,
        env_config: SiemensConfig,
        grasp_randomisation_x_range=(-0.001, 0.001),
        grasp_randomisation_z_range=(-0.001, 0.001),
        safety_box_radius=0.002,
        safety_box_step_size=0.0004,
        step_limit=150,
        minimal_start_goal_distance=0.0015,
        is_eval=False,
        use_pose_estimation=False,
        use_ft_controller: bool = True,
        use_6dof_grasp: bool = False,
    ):
        super().__init__(env)
        self.use_6dof_grasp = use_6dof_grasp
        self.action_space = spaces.Box(
            -np.inf, np.inf, (5,) if self.use_6dof_grasp else (2,)
        )
        self.alg_config = alg_config
        self.env_config = env_config
        self.home_config = env_config.custom_home_position
        self.grasp_position_ground_truth = env_config.grasp_position_ground_truth
        self.grasp_orientation_ground_truth = np.array(
            env_config.grasp_orientation_ground_truth_euler
        )
        self.goal_position_ground_truth = env_config.goal_position_ground_truth
        self.grasp_randomisation_x_range = grasp_randomisation_x_range
        self.grasp_randomisation_z_range = grasp_randomisation_z_range
        self.reset_grasp_delta = np.zeros(3)
        self.safety_box_radius = safety_box_radius
        self.safety_box_step_size = safety_box_step_size
        self.step_limit = step_limit
        self.n_since_last_home = 0
        self.first_reset = True
        self.n_steps = 0
        self.minimal_start_goal_distance = minimal_start_goal_distance
        self.is_eval = is_eval
        self.use_pose_estimation = use_pose_estimation
        self.use_ft_controller = use_ft_controller
        self.pose_estimation_helper = (
            PoseEstimationHelper(
                assumed_orientation=alg_config.pose_estimation_assumed_orientation,
                lock_orientation=not self.use_6dof_grasp,
            )
            if use_pose_estimation
            else None
        )
        self.pose_estimation_position_euler = np.array(
            alg_config.demo_goal_pose_estimation_euler
        )
        print("[InsertionWrapper] [__init__] Eval mode:", is_eval)

        self.ft_wrench_target = self.env_config.insertion_forcetorque
        self.x_force_k = self.env_config.ft_controller_k
        self.x_force_clip = 0.003
        self.i_term_clip = 0.0009

        # Per-episode FT log (inert by default — flip _ft_logging_active in
        # reset() if you want home-to-home logging in this wrapper too).
        self._episode_ft_log: list = []
        self._ft_logging_active: bool = False

        # Set by snap_push(reinforce=True) after it homes the robot itself,
        # so the next reset() can skip the redundant lift+waypoints+dropoff
        # sweep and the trailing home() call. Cleared by reset() once
        # consumed.
        self._already_homed_in_snap: bool = False

    def _logged_env_step(self, action):
        """Drop-in replacement for ``self._logged_env_step(action)`` that also
        appends the resulting FT reading to ``self._episode_ft_log`` when
        logging is active. Returns the same 5-tuple as the inner step."""
        out = self.env.step(action)
        if self._ft_logging_active:
            try:
                ft = out[0].get("observation.state.sensors_bota_ft_sensor")
                if ft is not None:
                    self._episode_ft_log.append(
                        np.asarray(ft, dtype=np.float64).copy()
                    )
            except Exception:
                pass
        if _abort_episode_flag.is_set():
            raise AbortEpisodeException("operator pressed 'u'")
        return out

    def pop_episode_ft_log(self) -> list:
        """Return the accumulated per-episode FT log and clear it."""
        log = self._episode_ft_log
        self._episode_ft_log = []
        return log

    def go_to_waypoint(
        self,
        current_obs,
        position,
        relative_pose_euler=None,
        distance_err=0.002,
        velocity_err=0.0005,
        is_via=True,
        is_rotated=False,
        coarse_follow_err_tol=0.002,
        coarse_velocity_tol=0.001,
        coarse_max_steps=600,
    ):
        relative_pose = (
            [0.0, 0.0, 0.0] if relative_pose_euler is None else relative_pose_euler
        )
        target = np.concatenate((position, relative_pose))
        obs, *_ = self._logged_env_step(
            target
            - np.concatenate(
                (current_obs["observation.state.cartesian"][:3], [0.0, 0.0, 0.0])
            )
        )

        # coarse — diagnostic prints + soft timeout so we never hang
        # silently. Thresholds and max-steps are caller-configurable so
        # non-critical waypoints (e.g. dropoff) can use looser tolerances
        # and exit fast.
        coarse_print_every = 50
        coarse_step = 0
        while (
            np.any(
                np.abs(
                    obs["observation.state.target"][:3]
                    - obs["observation.state.cartesian"][:3]
                )
                > coarse_follow_err_tol
            )
            or np.linalg.norm(obs["observation.velocity.cartesian"][:3])
            > coarse_velocity_tol
        ):
            obs, *_ = self._logged_env_step(np.zeros(6))
            coarse_step += 1
            if coarse_step % coarse_print_every == 0:
                follow_err = (
                    obs["observation.state.target"][:3]
                    - obs["observation.state.cartesian"][:3]
                )
                world_err = target[:3] - obs["observation.state.cartesian"][:3]
                vel = float(
                    np.linalg.norm(obs["observation.velocity.cartesian"][:3])
                )
                print(
                    f"  [go_to_waypoint] step {coarse_step}: "
                    f"follow_err(mm)={np.round(follow_err * 1000, 2)}  "
                    f"world_err(mm)={np.round(world_err * 1000, 2)}  "
                    f"|v|(mm/s)={vel * 1000:.2f}"
                )
            if coarse_step >= coarse_max_steps:
                follow_err = (
                    obs["observation.state.target"][:3]
                    - obs["observation.state.cartesian"][:3]
                )
                world_err = target[:3] - obs["observation.state.cartesian"][:3]
                print(
                    f"  [go_to_waypoint] TIMEOUT after {coarse_max_steps} steps. "
                    f"target={np.round(target[:3], 4)}  "
                    f"current={np.round(obs['observation.state.cartesian'][:3], 4)}  "
                    f"follow_err(mm)={np.round(follow_err * 1000, 2)}  "
                    f"world_err(mm)={np.round(world_err * 1000, 2)}.  "
                    f"Robot may be at a joint limit, singularity, or safety-box clip."
                )
                break
        # fine for terminal points
        if not is_via:
            err = target[:3] - obs["observation.state.cartesian"][:3]
            controller_error = (
                obs["observation.state.target"][:3]
                - obs["observation.state.cartesian"][:3]
            )
            while np.linalg.norm(err) > distance_err:
                # print(err, controller_error)
                i_term_clip = (
                    self.i_term_clip
                    if np.linalg.norm(obs["observation.velocity.cartesian"][:3]) > 0.001
                    else (
                        (
                            0.002
                            - np.linalg.norm(obs["observation.velocity.cartesian"][:3])
                        )
                        / 0.002
                        + 1
                    )
                    * self.i_term_clip
                )
                if np.any(np.abs(controller_error) > i_term_clip):
                    obs, *_ = self._logged_env_step(np.zeros(6))
                else:
                    obs, *_ = self._logged_env_step(
                        np.concatenate(
                            (
                                np.clip(
                                    err + controller_error,
                                    -i_term_clip,
                                    i_term_clip,
                                )
                                - controller_error,
                                np.zeros(3),
                            )
                        )
                    )
                err = target[:3] - obs["observation.state.cartesian"][:3]
                controller_error = (
                    obs["observation.state.target"][:3]
                    - obs["observation.state.cartesian"][:3]
                )
            while (
                np.linalg.norm(obs["observation.velocity.cartesian"][:3]) > velocity_err
            ):
                obs, *_ = self._logged_env_step(np.zeros(6))
                # print("waiting (fine)...")

        return obs

    def go_delta(
        self,
        current_obs,
        delta,
        relative_pose_euler=None,
        distance_err=0.002,
        velocity_err=0.0005,
        is_via=True,
    ):
        # Use actual cartesian position (not the controller's stale target) as
        # the reference. After an abort/truncation the controller's target can
        # be far ahead of where the robot actually is, which makes the
        # convergence target unreachable.
        target = delta + current_obs["observation.state.cartesian"][:3]
        relative_pose = (
            [0.0, 0.0, 0.0] if relative_pose_euler is None else relative_pose_euler
        )
        obs, *_ = self._logged_env_step(
            np.concatenate((delta, relative_pose))
        )

        # coarse — diagnostic prints + soft timeout (see go_to_waypoint)
        coarse_max_steps = 600
        coarse_print_every = 50
        coarse_step = 0
        while (
            np.linalg.norm(target - obs["observation.state.cartesian"][:3]) > 0.002
            or np.linalg.norm(obs["observation.velocity.cartesian"][:3]) > 0.001
        ):
            obs, *_ = self._logged_env_step(np.zeros(6))
            coarse_step += 1
            if coarse_step % coarse_print_every == 0:
                world_err = target - obs["observation.state.cartesian"][:3]
                vel = float(
                    np.linalg.norm(obs["observation.velocity.cartesian"][:3])
                )
                print(
                    f"  [go_delta] step {coarse_step}: "
                    f"world_err(mm)={np.round(world_err * 1000, 2)}  "
                    f"|v|(mm/s)={vel * 1000:.2f}"
                )
            if coarse_step >= coarse_max_steps:
                world_err = target - obs["observation.state.cartesian"][:3]
                print(
                    f"  [go_delta] TIMEOUT after {coarse_max_steps} steps. "
                    f"target={np.round(target, 4)}  "
                    f"current={np.round(obs['observation.state.cartesian'][:3], 4)}  "
                    f"world_err(mm)={np.round(world_err * 1000, 2)}.  "
                    f"Robot may be at a joint limit, singularity, or safety-box clip."
                )
                break
        # fine for terminal points
        if not is_via:
            err = target[:3] - obs["observation.state.cartesian"][:3]
            controller_error = (
                obs["observation.state.target"][:3]
                - obs["observation.state.cartesian"][:3]
            )
            while np.linalg.norm(err) > distance_err:
                if np.any(np.abs(controller_error) > self.i_term_clip):
                    obs, *_ = self._logged_env_step(np.zeros(6))
                else:
                    obs, *_ = self._logged_env_step(
                        np.concatenate(
                            (
                                np.clip(
                                    err + controller_error,
                                    -self.i_term_clip,
                                    self.i_term_clip,
                                )
                                - controller_error,
                                np.zeros(3),
                            )
                        )
                    )
                err = target[:3] - obs["observation.state.cartesian"][:3]
                controller_error = (
                    obs["observation.state.target"][:3]
                    - obs["observation.state.cartesian"][:3]
                )
            while (
                np.linalg.norm(obs["observation.velocity.cartesian"][:3]) > velocity_err
            ):
                obs, *_ = self._logged_env_step(np.zeros(6))

        return obs

    def yz_i_controller_dyz(self, obs, target):
        err = target[1:3] - obs["observation.state.cartesian"][1:3]
        controller_error = (
            obs["observation.state.target"][1:3]
            - obs["observation.state.cartesian"][1:3]
        )
        if np.any(np.abs(controller_error) > self.i_term_clip):
            return np.zeros(2)
        else:
            return (
                np.clip(
                    err + controller_error,
                    -self.i_term_clip,
                    self.i_term_clip,
                )
                - controller_error
            )

    def x_torque_controller_dx(self, obs):
        y_torque_sensed = np.sum(
            obs["observation.state.sensors_bota_ft_sensor"][3:5]
        ) / np.sqrt(2)
        y_torque_error = self.ft_wrench_target - y_torque_sensed
        x_force_error = y_torque_error / self.env_config.ft_controller_lever_arm
        x_impedance_error = (
            obs["observation.state.target"][0] - obs["observation.state.cartesian"][0]
        )
        if x_force_error > 0 and x_impedance_error < self.x_force_clip:
            dx = -min(
                x_force_error / self.x_force_k, self.x_force_clip - x_impedance_error
            )

        elif x_force_error < 0 and x_impedance_error > -self.x_force_clip:
            dx = -max(
                x_force_error / self.x_force_k, -self.x_force_clip - x_impedance_error
            )
        else:
            dx = 0.0
        # print(
        #     f"[X torque controller] target: {self.ft_wrench_target}, sensed: {y_torque_sensed:.3f}, force_error: {x_force_error:.3f}, impedance_error: {x_impedance_error:.5f}, dx: {dx:.5f}"
        # )
        return dx, x_force_error

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> tuple[Any, dict[str, Any]]:
        # Consume the snap-push-homed flag (set when snap_push(reinforce=True)
        # already drove the robot to home + opened the gripper). When True
        # the lift/waypoints/dropoff sweep AND the subsequent home() call
        # are both redundant — skip them and go straight to env.reset().
        skip_reset_sweep = self._already_homed_in_snap
        self._already_homed_in_snap = False

        if not self.first_reset:
            if skip_reset_sweep:
                print(
                    "Skipping reset sweep (snap_push already homed and opened "
                    "the gripper)."
                )
            else:
                # lift up
                self.obs, *_ = self._logged_env_step(np.zeros(6))  # wait one step
                self.obs = self.go_delta(
                    self.obs,
                    self.env_config.relative_motion_after_rl_train[:3],
                    self.env_config.relative_motion_after_rl_train[3:],
                )
                # waypoints back
                for pose, res in self.env_config.waypoints_after_rl_train:
                    self.obs = self.go_to_waypoint(
                        self.obs,
                        pose[:3],
                        pose[3:],
                        distance_err=res,
                    )
                # dropoff location — non-critical, just need to be roughly
                # over the dropoff before opening the gripper. Use loose tols
                # and a short step budget so we don't burn 40 s waiting for
                # sub-mm convergence here.
                self.obs = self.go_to_waypoint(
                    self.obs,
                    self.env_config.dropoff_point,
                    coarse_follow_err_tol=0.005,  # 5 mm
                    coarse_velocity_tol=0.005,    # 5 mm/s
                    coarse_max_steps=50,
                )
                # Drop the brick at the dropoff point before going home,
                # rather than carrying it through the home sweep.
                print("Opening gripper at dropoff...")
                self.env.unwrapped.gripper.set_target(0.85)  # type: ignore
                time.sleep(0.8)

        else:
            print("homing first time...")
            # Robust safety lift on first reset (works from gravity-comp).
            # We can't use go_delta here: the cartesian controller is not
            # engaged yet and robot.target_pose is uninitialised, so
            # go_delta's convergence loop hangs forever.
            #
            # Instead, use manipulator_env.move_to(), which is self-contained:
            #   (a) reads the current EE pose from a ROS topic (does NOT
            #       require an active controller),
            #   (b) switches to the cartesian controller,
            #   (c) opens the gripper,
            #   (d) runs a planned trajectory at `speed` m/s,
            #   (e) resets robot.target_pose to the reached pose,
            #   (f) switches back to the default controller.
            self.env.unwrapped.robot.wait_until_ready()  # type: ignore
            current_xyz = np.array(
                self.env.unwrapped.robot.end_effector_pose.position  # type: ignore
            )
            #target_xyz = current_xyz + np.array([0.0, 0.0, 0.040])
            target_xyz = current_xyz + np.array([0.0, 0.0, 0.000])
            print(f"  lift 40 mm:  {current_xyz}  ->  {target_xyz}")
            try:
                self.env.unwrapped.move_to(position=target_xyz, speed=0.03)  # type: ignore
            except Exception as e:
                # Don't trap the user at "homing first time..." if the lift
                # fails (unreachable target, controller refused, etc.) —
                # fall through to the home() call below, which uses the
                # joint trajectory controller and is the original behaviour.
                print(f"  WARNING: move_to lift failed ({e!r}); skipping lift.")
            print("  homing to custom_first_home_position...")
            self.env.unwrapped.home(  # type: ignore
                home_config=self.env_config.custom_first_home_position
            )

            self.first_reset = False
        if skip_reset_sweep:
            print("Skipping homing (snap_push already drove robot to home).")
        else:
            print("homing...")
            self.env.unwrapped.home(home_config=self.home_config)  # type: ignore

        if options is not None and options.get("last_reset", False):
            print("Last reset, not going to start position.")
            return self.obs, {}

        self.obs, reset_info = self.env.reset(seed=seed, options=options)

        self.target_grasp_position = np.copy(self.grasp_position_ground_truth)

        grasp_randomisation_x = np.random.uniform(
            self.grasp_randomisation_x_range[0], self.grasp_randomisation_x_range[1]
        )
        grasp_randomisation_z = np.random.uniform(
            self.grasp_randomisation_z_range[0], self.grasp_randomisation_z_range[1]
        )

        self.target_grasp_position[0] += grasp_randomisation_x
        self.target_grasp_position[2] += grasp_randomisation_z

        # print("Executing before-grasp motion")
        # self.obs, *_ = self._logged_env_step(self.env_config.relative_motion_before_grasp)
        print("Opening gripper...")
        self.env.unwrapped.gripper.set_target(0.85)  # type: ignore
        time.sleep(1.5)

        print("Moving to grasp position...")
        self.obs = self.go_to_waypoint(
            self.obs,
            self.target_grasp_position,
            self.grasp_orientation_ground_truth if self.use_6dof_grasp else None,
            distance_err=0.0002,
            is_via=False,
        )
        print("Grasping...")
        self.env.unwrapped.gripper.set_target(0.65)  # type: ignore
        time.sleep(2.2)
        self.obs, *_ = self._logged_env_step(np.zeros(6))
        self.actual_grasp_position = np.copy(
            self.obs["observation.state.cartesian"][:3]
        )
        self.reset_grasp_delta = (
            self.actual_grasp_position - self.grasp_position_ground_truth
        )

        # pick up quickly
        print("Picking up...")
        self.obs = self.go_delta(
            self.obs,
            self.env_config.relative_motion_after_grasp[:3],
            self.grasp_orientation_ground_truth if self.use_6dof_grasp else None,
        )

        # compute goal position
        if self.use_pose_estimation and self.pose_estimation_helper is not None:
            # go to pose estimation position; estimate; compare to demo pose estimation; compute goal position
            raise RuntimeError("Not yet implemented")
            print("Moving to pose estimation position...")
            self.obs = self.go_to_cartesian(
                self.obs,
                target_cartesian=self.pose_estimation_position_euler[:3],
                fine_resolution=0.0005,
            )
            self.obs, *_ = self._logged_env_step(np.zeros(6))
            self.actual_estimation_position = np.copy(
                self.obs["observation.state.cartesian"]
            )
            pose_estimation_joint_state = np.copy(self.obs["observation.state.joints"])
            print("Estimating pose...")
            lavender_pose, purple_pose = (
                self.pose_estimation_helper.estimate_two_lego_bricks_absolute(
                    self.obs["observation.images.wrist_camera"],
                    self.obs["observation.images.wrist_depth_camera"],
                    self.actual_estimation_position,
                )
            )

            self.goal_position = (
                self.actual_estimation_position[:3]
                + purple_pose[:3, 3]
                - lavender_pose[:3, 3]
            )

        else:
            self.goal_position = np.copy(self.goal_position_ground_truth)
            self.estimated_grasp_delta = np.copy(self.reset_grasp_delta)
            self.estimated_grasp_delta[0] += np.random.uniform(
                *self.grasp_randomisation_x_range
            )
            self.estimated_grasp_delta[2] += np.random.uniform(
                *self.grasp_randomisation_z_range
            )

            self.goal_position += self.estimated_grasp_delta

        if not self.is_eval:
            corrected_goal_gt = self.goal_position_ground_truth + self.reset_grasp_delta
            self.start_position = corrected_goal_gt.copy()
            while np.linalg.norm(
                self.start_position[1:3] - corrected_goal_gt[1:3]
            ) < self.minimal_start_goal_distance or np.any(
                np.clip(
                    self.start_position[1:3] - corrected_goal_gt[1:3],
                    [-0.004, -0.001],
                    [0.004, 0.003],
                )
                != self.start_position[1:3] - corrected_goal_gt[1:3]
            ):
                self.start_position[1:3] = self.goal_position[1:3] + np.random.uniform(
                    -self.safety_box_radius, self.safety_box_radius, size=2
                )
            with printoptions(precision=4):
                print(
                    f"goal true (corr) pos: {self.goal_position_ground_truth + self.reset_grasp_delta}, goal est pos {self.goal_position}, start pos {self.start_position}"
                )
        else:
            self.start_position = self.goal_position.copy()

        for pose, res in self.env_config.waypoints_after_grasp:
            self.obs = self.go_to_waypoint(
                self.obs,
                pose[:3] + self.estimated_grasp_delta,
                pose[3:],
                distance_err=res,
            )

        # lower down slowly until z-force is established in steps of 3mm
        if self.use_ft_controller:
            print("Establishing contact...")
            self.obs, *_ = self._logged_env_step(np.zeros(6))
            self.env.tare_ft_sensor(self.obs)  # pyright: ignore[reportAttributeAccessIssue]
            # [s.reset() for s in self.env.unwrapped.sensors]  # pyright: ignore[reportAttributeAccessIssue] # tare ft sensor
            x_step, x_force_error = self.x_torque_controller_dx(self.obs)
            while abs(x_force_error) > 0.1:  # wait until some contact
                # delta_yz = self.yz_i_controller_dyz(self.obs, self.start_position[1:3])
                delta_yz = (
                    np.clip(
                        self.start_position[1:3]
                        - self.obs["observation.state.cartesian"][1:3],
                        -self.i_term_clip,
                        self.i_term_clip,
                    )
                    if np.linalg.norm(self.obs["observation.velocity.cartesian"]) < 0.001
                    else np.zeros(2)
                )
                self.obs, *_ = self._logged_env_step(
                    np.array([x_step, delta_yz[0], delta_yz[1], 0, 0, 0])
                )
                x_step, x_force_error = self.x_torque_controller_dx(self.obs)
        else:
            print("Skipping contact establishment (FT controller disabled).")

        self.n_steps = 0
        self.obs, reset_info = self.env.reset()
        reset_info["reset.grasped.position"] = self.actual_grasp_position

        reset_info["reset.grasped.delta"] = self.reset_grasp_delta
        reset_info["reset.grasped.delta_estimated"] = self.estimated_grasp_delta
        goal_position_offset = self.goal_position - (
            self.goal_position_ground_truth + self.reset_grasp_delta
        )
        reset_info["reset.goal_position.offset"] = goal_position_offset
        if self.use_pose_estimation:
            reset_info["reset.pose_estimation.joint_state"] = (
                pose_estimation_joint_state  # pyright: ignore[reportUndefinedVariable]
            )
            reset_info["reset.pose_estimation.cartesian"] = (
                self.actual_estimation_position
            )  # pyright: ignore[reportPossiblyUnboundVariable]
        with printoptions(precision=4):
            print(
                f"goal true (corr) pos: {self.goal_position_ground_truth + self.reset_grasp_delta}, goal est pos {self.goal_position}, start pos {self.start_position}, reached {self.obs['observation.state.cartesian'][:3]}"
            )
        print("Reset complete.")
        self.obs = self.add_perfect_action_to_obs(self.obs)
        return self.obs, reset_info

    # ------------------------------------------------------------------ #
    # snap_push (called by SuccessClassificationWrapper on E_SUCCESS_CLS)  #
    # ------------------------------------------------------------------ #
    def _step_translation_clipped(self, dxyz: np.ndarray) -> np.ndarray:
        """Inline 6-dim action helper for the timed reinforce inner loops."""
        action6 = np.array(
            [dxyz[0], dxyz[1], dxyz[2], 0.0, 0.0, 0.0], dtype=np.float64
        )
        out, *_ = self._logged_env_step(action6)
        return out

    def _settle_for(self, seconds: float) -> None:
        """Equivalent of time.sleep(seconds) but keeps stepping zero actions
        so the cartesian controller can converge to its current target.
        Plain time.sleep blocks the Python thread without sending any
        env.step, so the controller hits the next motion block mid-trajectory
        and the robot 'keeps moving' after the sleep."""
        t0 = time.time()
        while time.time() - t0 < seconds:
            self.obs, *_ = self._logged_env_step(np.zeros(6))

    def snap_push(
        self,
        push_distance: float = 0.003,
        pause_before: float = 1.0,
        pause_after: float = 1.0,
        reinforce: bool = False,
        reinforce_lift: float = 0.013,
        reinforce_press: float = 0.010,
        reinforce_post_lift: float = 0.030,
        reinforce_push_offset_x: float = -0.020,
    ) -> None:
        """Seat the part after the success classifier fires. Mirrors the
        3DoFRotZPE wrapper's snap_push:

        Phase 1 (always): pause, press -Z by push_distance, hold.
        Phase 2 (reinforce=True): open gripper → lift +Z → close gripper →
        press -Z → lift +Z → open → descend → re-grasp.

        XY anchoring uses the ground-truth goal corrected by the measured
        grasp delta (analog of the PE variant's goal_position anchor)."""
        
        if not reinforce:
            return

        # ---- reinforce: open / lift / close / press / lift / open / descend / re-grasp ----
        snap_pos = self.obs["observation.state.cartesian"][:3].copy()
        # Anchor XY on the corrected goal (where the lid should be) so the
        # regrasp lands on-center even if the policy terminated a few mm off.
        goal_xy = np.asarray(
            (self.goal_position_ground_truth + self.reset_grasp_delta)[:2],
            dtype=np.float64,
        ).copy()
        regrasp_target = np.array(
            [float(goal_xy[0]), float(goal_xy[1]), float(snap_pos[2])]
        )
        print(
            f"[InsertionWrapperSiemens] snap_push: reinforce — anchor "
            f"goal_xy=[{goal_xy[0]*1e3:.1f}, {goal_xy[1]*1e3:.1f}] mm  "
            f"snap_z={snap_pos[2]*1e3:.1f} mm"
        )

        # Anchor XY for the press cycle: shifted by reinforce_push_offset_x in
        # X relative to the goal. ALL subsequent Z motions reuse this XY so
        # the gripper never moves laterally over the placed lid while
        # pressing.
        shifted_xy = np.array(
            [float(goal_xy[0]) + reinforce_push_offset_x, float(goal_xy[1])]
        )
        snap_z = float(snap_pos[2])

        print("[InsertionWrapperSiemens] snap_push: reinforce — opening gripper...")
        self.env.unwrapped.gripper.set_target(0.94)  # type: ignore
        self._settle_for(0.8)

        # Steps 2 + 3 combined: shift X and lift +Z in one diagonal move to
        # the lift1 target. Subsequent press/lift2 reuse shifted_xy so X
        # stays put through the press cycle.
        print(
            f"[InsertionWrapperSiemens] snap_push: reinforce — shifting X by "
            f"{reinforce_push_offset_x * 1e3:+.1f} mm and lifting "
            f"{reinforce_lift*1e3:.1f} mm in Z..."
        )
        lift1_target = np.array([shifted_xy[0], shifted_xy[1], snap_z + reinforce_lift])
        t0 = time.time()
        while time.time() - t0 < 3.6:
            err = lift1_target - self.obs["observation.state.cartesian"][:3]
            self.obs = self._step_translation_clipped(
                np.clip(err, -self.i_term_clip, self.i_term_clip)
            )

        print("[InsertionWrapperSiemens] snap_push: reinforce — closing gripper...")
        self.env.unwrapped.gripper.set_target(0.6)  # type: ignore
        self._settle_for(1.2)

        # Step 5: press -Z (XY stays at shifted_xy).
        # End Z = snap_z + reinforce_lift - reinforce_press
        print(
            f"[InsertionWrapperSiemens] snap_push: reinforce — pressing down "
            f"{reinforce_press*1e3:.1f} mm in Z..."
        )
        press_target = np.array(
            [shifted_xy[0], shifted_xy[1], snap_z + reinforce_lift - reinforce_press]
        )
        t0 = time.time()
        while time.time() - t0 < 1.4:
            err = press_target - self.obs["observation.state.cartesian"][:3]
            self.obs = self._step_translation_clipped(
                np.clip(err, -self.i_term_clip, self.i_term_clip)
            )

        # Step 6: lift +Z again (XY stays at shifted_xy).
        # End Z = press_z + reinforce_post_lift = snap_z + lift - press + post_lift
        print(
            f"[InsertionWrapperSiemens] snap_push: reinforce — lifting "
            f"{reinforce_post_lift*1e3:.1f} mm in Z..."
        )
        lift2_target = np.array(
            [
                shifted_xy[0],
                shifted_xy[1],
                snap_z + reinforce_lift - reinforce_press + reinforce_post_lift,
            ]
        )
        t0 = time.time()
        while time.time() - t0 < 1.4:
            err = lift2_target - self.obs["observation.state.cartesian"][:3]
            self.obs = self._step_translation_clipped(
                np.clip(err, -self.i_term_clip*2, self.i_term_clip*2)
            )

        print("[InsertionWrapperSiemens] snap_push: reinforce — opening gripper...")
        self.env.unwrapped.gripper.set_target(0.94)  # type: ignore
        # self._settle_for(0.4)
        
        print("[InsertionWrapperSiemens] snap_push: reinforce — lifting +50 mm in Z before homing...")
        self.obs = self.go_delta(self.obs, [0.0, 0.030, 0.030])

        print("[InsertionWrapperSiemens] snap_push: reinforce — waiting 3 s before homing...")
        self._settle_for(2.0)

        print("[InsertionWrapperSiemens] snap_push: reinforce — going home...")
        self.env.unwrapped.home(home_config=self.home_config)  # type: ignore
        self._already_homed_in_snap = True
        print("[InsertionWrapperSiemens] snap_push: reinforce done.")

    def step(self, action) -> tuple[Any, Any, bool, bool, dict[str, Any]]:
        # action_input = np.copy(action)
        x_action = (
            self.x_torque_controller_dx(self.obs)[0] if self.use_ft_controller else 0.0
        )
        if self.use_6dof_grasp:
            if action.shape[0] < 5:
                raise ValueError(
                    f"Expected 5D action in 6DoF mode, got shape {action.shape}"
                )
            action = np.array(
                [x_action, action[0], action[1], action[2], action[3], action[4]]
            )
        else:
            action = np.array([x_action, action[0], action[1], 0, 0, 0])

        # apply safety box
        current_pos_yz = self.obs["observation.state.cartesian"][1:3]
        delta_yz = current_pos_yz - self.goal_position[1:3]
        delta_yz_clipped = np.clip(
            delta_yz, -self.safety_box_radius, self.safety_box_radius
        )
        if np.any(delta_yz_clipped != delta_yz):
            correcting_action = delta_yz_clipped - delta_yz
            action[1:3] = correcting_action

        # print(f"[Step] received {action_input}, performing {action}")
        self.obs, reward, terminated, truncated, info = self._logged_env_step(action)
        self.obs = self.add_perfect_action_to_obs(self.obs)

        self.n_steps += 1
        if self.n_steps >= self.step_limit:
            truncated = True

        return self.obs, reward, terminated, truncated, info

    def add_perfect_action_to_obs(self, obs):
        obs["observation.perfect_action"] = (
            self.goal_position_ground_truth + self.reset_grasp_delta
        ) - obs["observation.state.cartesian"][:3]
        return obs


class InsertionWrapperSiemensPE(Wrapper):
    def __init__(
        self,
        env,
        alg_config: Config,
        env_config: SiemensConfig,
        safety_box_radius=0.003,
        safety_box_step_size=0.0004,
        step_limit=150,
        use_ft_controller: bool = True,
        use_6dof_grasp: bool = False,
        pe_align_6dof: bool = False,
        pose_viz_dir: Optional[str] = None,
    ):
        super().__init__(env)
        self.use_ft_controller = use_ft_controller
        self.use_6dof_grasp = use_6dof_grasp
        # Decoupled flag: whether PE/grasp alignment uses full 6DoF.
        # - use_6dof_grasp=True implies 6DoF alignment (action space is 5D too).
        # - pe_align_6dof=True with use_6dof_grasp=False gives 6DoF alignment
        #   for the grasp + standard post-grasp "undo" back to demo neutral,
        #   while keeping the policy action space at 2D — useful for running
        #   a 2D-action policy on a tilted brick.
        self._pe_alignment_is_6dof = bool(use_6dof_grasp or pe_align_6dof)
        self.pose_viz_dir = pose_viz_dir
        self._pe_episode_idx = -1
        self._pose_overlay_renderer = None
        if pose_viz_dir is not None:
            from crisp_drl.envs.pose_visualizer import PoseOverlayRenderer
            self._pose_overlay_renderer = PoseOverlayRenderer(
                camera_info_json_path="camera_parameters/realsense_d405_single.json",
            )
            print(f"[InsertionWrapperSiemensPE] pose viz dir = {pose_viz_dir}")
        self.action_space = spaces.Box(
            -np.inf, np.inf, (5,) if self.use_6dof_grasp else (2,)
        )
        self.alg_config = alg_config
        self.env_config = env_config
        self.home_config = env_config.custom_home_position_pe
        self.grasp_position_ground_truth = env_config.grasp_position_ground_truth
        self.grasp_orientation_ground_truth = np.array(
            env_config.grasp_orientation_ground_truth_euler
        )
        self.goal_position_ground_truth = env_config.goal_position_ground_truth
        self.pose_estimation_helper = PoseEstimationHelper(
            assumed_orientation=np.array([])
        )
        self.safety_box_radius = safety_box_radius
        self.safety_box_step_size = safety_box_step_size
        self.step_limit = step_limit
        self.n_since_last_home = 0
        self.first_reset = True
        self.n_steps = 0
        self.pose_estimation_position_euler = np.array(
            alg_config.demo_goal_pose_estimation_euler
        )

        self.ft_wrench_target = self.env_config.insertion_forcetorque
        self.x_force_k = self.env_config.ft_controller_k
        self.z_force_k = self.env_config.ft_controller_k
        self.x_force_clip = 0.003
        self.z_force_clip = 0.003
        self.i_term_clip = 0.0012

        # Per-episode FT log captured by _logged_env_step (one sample per
        # internal env.step), spanning the ENTIRE episode — reset (grasp,
        # PE, contact) + policy steps + snap_push. Eval reads via
        # pop_episode_ft_log() after termination.
        self._episode_ft_log: list = []
        self._ft_logging_active: bool = False

        # Set by snap_push(reinforce=True) after it homes the robot itself,
        # so the next reset() can skip the redundant lift+waypoints+dropoff
        # sweep and the trailing home() call. Cleared by reset() once
        # consumed.
        self._already_homed_in_snap: bool = False

        self.o_T_o_tcpgrasp = self.env_config.demo_w_D_w_o[:3, :3].T @ (
            self.grasp_position_ground_truth - self.env_config.demo_w_D_w_o[:3, 3]
        )
        self.pose_estimation_settle_steps = 1
        self.alignment_clip_angle_rad = np.deg2rad(25.0)
        self.alignment_pitch_bias = np.deg2rad(-3.0)
        print(
            "[InsertionWrapperSiemensPE] use_6dof_grasp=",
            self.use_6dof_grasp,
            "pe_alignment_6dof=",
            self._pe_alignment_is_6dof,
            "action_dim=",
            self.action_space.shape[0],
        )

    def _logged_env_step(self, action):
        """Drop-in replacement for ``self._logged_env_step(action)`` that also
        appends the resulting FT reading to ``self._episode_ft_log`` when
        logging is active. Returns the same 5-tuple as the inner step."""
        out = self.env.step(action)
        if self._ft_logging_active:
            try:
                ft = out[0].get("observation.state.sensors_bota_ft_sensor")
                if ft is not None:
                    self._episode_ft_log.append(
                        np.asarray(ft, dtype=np.float64).copy()
                    )
            except Exception:
                pass
        if _abort_episode_flag.is_set():
            raise AbortEpisodeException("operator pressed 'u'")
        return out

    def pop_episode_ft_log(self) -> list:
        """Return the accumulated per-episode FT log and clear it."""
        log = self._episode_ft_log
        self._episode_ft_log = []
        return log

    def safe_abort(self) -> None:
        """Called by the eval runner after the operator presses 'u'. Brings the
        robot to a safe state (lift, open gripper, home) and sets the
        skip-reset-sweep flag so reset() does not run the re-grasp+press-down
        sequence on the next episode."""
        print("[InsertionWrapperSiemensPE] safe_abort — opening gripper...")
        try:
            self.env.unwrapped.gripper.set_target(0.85)  # type: ignore
        except Exception as e:
            print(f"[InsertionWrapperSiemensPE] safe_abort: gripper open failed: {e}")
        time.sleep(1.0)
        print("[InsertionWrapperSiemensPE] safe_abort — +Y +Z 30 mm lift...")
        try:
            self.obs = self.go_delta(self.obs, [0.0, 0.030, 0.030])
        except Exception as e:
            print(f"[InsertionWrapperSiemensPE] safe_abort: lift failed: {e}")
        print("[InsertionWrapperSiemensPE] safe_abort — homing...")
        try:
            self.env.unwrapped.home(home_config=self.home_config)  # type: ignore
        except Exception as e:
            print(f"[InsertionWrapperSiemensPE] safe_abort: home failed: {e}")
        self._already_homed_in_snap = True
        print("[InsertionWrapperSiemensPE] safe_abort done.")

    def compute_alignment_rpy(self, world_D_world_obj: np.ndarray) -> np.ndarray:
        world_R_demoobj = self.env_config.demo_w_D_w_o[:3, :3]
        world_R_estiobj = world_D_world_obj[:3, :3]
        est_R_demo = world_R_estiobj @ world_R_demoobj.T

        # In non-6DoF mode, take yaw only (open-loop yaw alignment).
        # `_pe_alignment_is_6dof` is set by either --use_6dof_grasp OR
        # --pe_align_6dof, so the alignment can be 6DoF even with a 2D policy.
        if not self._pe_alignment_is_6dof:
            yaw_only = np.array(
                [0.0, 0.0, rot_matrix_to_euler_xyz(est_R_demo)[2]]
            )
            yaw_only[2] = float(
                np.clip(
                    yaw_only[2],
                    -self.alignment_clip_angle_rad,
                    self.alignment_clip_angle_rad,
                ) 
            )
            return yaw_only

        # 6DoF: apply hardcoded pitch bias in the demo->estimated relative frame,
        # then clip the *total* rotation magnitude (axis-angle) so the bound is
        # meaningful under ZYX coupling and applies uniformly to all components.
        from scipy.spatial.transform import Rotation as _R

        bias_R = euler_to_rot_matrix(0.0, self.alignment_pitch_bias, 0.0)
        biased = est_R_demo @ bias_R
        rotvec = _R.from_matrix(biased).as_rotvec()
        angle = float(np.linalg.norm(rotvec))
        if angle > self.alignment_clip_angle_rad and angle > 0.0:
            rotvec = rotvec * (self.alignment_clip_angle_rad / angle)
            biased = _R.from_rotvec(rotvec).as_matrix()
        return rot_matrix_to_euler_xyz(biased)

    def estimate_world_pose_once(self, return_details: bool = False):
        for _ in range(self.pose_estimation_settle_steps):
            self.obs, *_ = self._logged_env_step(np.zeros(6))
        return self.pose_estimation_helper.estimate_siemens_world_frame_coarse(
            self.obs["observation.images.wrist_camera"],
            self.obs["observation.images.wrist_depth_camera"],
            self.obs["observation.state.cartesian"],
            return_details=return_details,
        )

    def validate_world_pose_transform(
        self, world_D_world_obj: np.ndarray, tag: str
    ) -> dict[str, float | bool]:
        rot = world_D_world_obj[:3, :3]
        finite_ok = bool(np.isfinite(world_D_world_obj).all())
        orth_err = float(np.linalg.norm(rot.T @ rot - np.eye(3), ord="fro"))
        det_r = float(np.linalg.det(rot))
        valid = finite_ok and abs(det_r - 1.0) < 1e-2 and orth_err < 1e-2
        print(
            f"[InsertionWrapperSiemensPE] {tag} transform valid={valid} "
            f"det={det_r:.6f} orth_err={orth_err:.6e}"
        )
        return {"valid": valid, "det_r": det_r, "orth_err": orth_err}

    def go_to_waypoint(
        self,
        current_obs,
        position,
        relative_pose_euler=None,
        distance_err=0.002,
        velocity_err=0.0005,
        is_via=True,
        is_rotated=False,
        coarse_follow_err_tol=0.002,
        coarse_velocity_tol=0.001,
        coarse_max_steps=600,
    ):
        relative_pose = (
            [0.0, 0.0, 0.0] if relative_pose_euler is None else relative_pose_euler
        )
        target = np.concatenate((position, relative_pose))
        obs, *_ = self._logged_env_step(
            target
            - np.concatenate(
                (current_obs["observation.state.cartesian"][:3], [0.0, 0.0, 0.0])
            )
        )

        # coarse — diagnostic prints + soft timeout so we never hang
        # silently. Thresholds and max-steps are caller-configurable so
        # non-critical waypoints (e.g. dropoff) can use looser tolerances
        # and exit fast.
        coarse_print_every = 50
        coarse_step = 0
        while (
            np.any(
                np.abs(
                    obs["observation.state.target"][:3]
                    - obs["observation.state.cartesian"][:3]
                )
                > coarse_follow_err_tol
            )
            or np.linalg.norm(obs["observation.velocity.cartesian"][:3])
            > coarse_velocity_tol
        ):
            obs, *_ = self._logged_env_step(np.zeros(6))
            coarse_step += 1
            if coarse_step % coarse_print_every == 0:
                follow_err = (
                    obs["observation.state.target"][:3]
                    - obs["observation.state.cartesian"][:3]
                )
                world_err = target[:3] - obs["observation.state.cartesian"][:3]
                vel = float(
                    np.linalg.norm(obs["observation.velocity.cartesian"][:3])
                )
                print(
                    f"  [go_to_waypoint] step {coarse_step}: "
                    f"follow_err(mm)={np.round(follow_err * 1000, 2)}  "
                    f"world_err(mm)={np.round(world_err * 1000, 2)}  "
                    f"|v|(mm/s)={vel * 1000:.2f}"
                )
            if coarse_step >= coarse_max_steps:
                follow_err = (
                    obs["observation.state.target"][:3]
                    - obs["observation.state.cartesian"][:3]
                )
                world_err = target[:3] - obs["observation.state.cartesian"][:3]
                print(
                    f"  [go_to_waypoint] TIMEOUT after {coarse_max_steps} steps. "
                    f"target={np.round(target[:3], 4)}  "
                    f"current={np.round(obs['observation.state.cartesian'][:3], 4)}  "
                    f"follow_err(mm)={np.round(follow_err * 1000, 2)}  "
                    f"world_err(mm)={np.round(world_err * 1000, 2)}.  "
                    f"Robot may be at a joint limit, singularity, or safety-box clip."
                )
                break
        # fine for terminal points
        if not is_via:
            err = target[:3] - obs["observation.state.cartesian"][:3]
            controller_error = (
                obs["observation.state.target"][:3]
                - obs["observation.state.cartesian"][:3]
            )
            while np.linalg.norm(err) > distance_err:
                # print(err, controller_error)
                i_term_clip = (
                    self.i_term_clip
                    if np.linalg.norm(obs["observation.velocity.cartesian"][:3]) > 0.001
                    else (
                        (
                            0.002
                            - np.linalg.norm(obs["observation.velocity.cartesian"][:3])
                        )
                        / 0.002
                        + 1
                    )
                    * self.i_term_clip
                )
                if np.any(np.abs(controller_error) > i_term_clip):
                    obs, *_ = self._logged_env_step(np.zeros(6))
                else:
                    obs, *_ = self._logged_env_step(
                        np.concatenate(
                            (
                                np.clip(
                                    err + controller_error,
                                    -i_term_clip,
                                    i_term_clip,
                                )
                                - controller_error,
                                np.zeros(3),
                            )
                        )
                    )
                err = target[:3] - obs["observation.state.cartesian"][:3]
                controller_error = (
                    obs["observation.state.target"][:3]
                    - obs["observation.state.cartesian"][:3]
                )
            while (
                np.linalg.norm(obs["observation.velocity.cartesian"][:3]) > velocity_err
            ):
                obs, *_ = self._logged_env_step(np.zeros(6))
                # print("waiting (fine)...")

        return obs

    def go_delta(
        self,
        current_obs,
        delta,
        relative_pose_euler=None,
        distance_err=0.002,
        velocity_err=0.0005,
        is_via=True,
    ):
        # Use actual cartesian position (not the controller's stale target) as
        # the reference. After an abort/truncation the controller's target can
        # be far ahead of where the robot actually is, which makes the
        # convergence target unreachable.
        target = delta + current_obs["observation.state.cartesian"][:3]
        relative_pose = (
            [0.0, 0.0, 0.0] if relative_pose_euler is None else relative_pose_euler
        )
        obs, *_ = self._logged_env_step(
            np.concatenate((delta, relative_pose))
        )

        # coarse — diagnostic prints + soft timeout (see go_to_waypoint)
        coarse_max_steps = 600
        coarse_print_every = 50
        coarse_step = 0
        while (
            np.linalg.norm(target - obs["observation.state.cartesian"][:3]) > 0.002
            or np.linalg.norm(obs["observation.velocity.cartesian"][:3]) > 0.001
        ):
            obs, *_ = self._logged_env_step(np.zeros(6))
            coarse_step += 1
            if coarse_step % coarse_print_every == 0:
                world_err = target - obs["observation.state.cartesian"][:3]
                vel = float(
                    np.linalg.norm(obs["observation.velocity.cartesian"][:3])
                )
                print(
                    f"  [go_delta] step {coarse_step}: "
                    f"world_err(mm)={np.round(world_err * 1000, 2)}  "
                    f"|v|(mm/s)={vel * 1000:.2f}"
                )
            if coarse_step >= coarse_max_steps:
                world_err = target - obs["observation.state.cartesian"][:3]
                print(
                    f"  [go_delta] TIMEOUT after {coarse_max_steps} steps. "
                    f"target={np.round(target, 4)}  "
                    f"current={np.round(obs['observation.state.cartesian'][:3], 4)}  "
                    f"world_err(mm)={np.round(world_err * 1000, 2)}.  "
                    f"Robot may be at a joint limit, singularity, or safety-box clip."
                )
                break
        # fine for terminal points
        if not is_via:
            err = target[:3] - obs["observation.state.cartesian"][:3]
            controller_error = (
                obs["observation.state.target"][:3]
                - obs["observation.state.cartesian"][:3]
            )
            while np.linalg.norm(err) > distance_err:
                if np.any(np.abs(controller_error) > self.i_term_clip):
                    obs, *_ = self._logged_env_step(np.zeros(6))
                else:
                    obs, *_ = self._logged_env_step(
                        np.concatenate(
                            (
                                np.clip(
                                    err + controller_error,
                                    -self.i_term_clip,
                                    self.i_term_clip,
                                )
                                - controller_error,
                                np.zeros(3),
                            )
                        )
                    )
                err = target[:3] - obs["observation.state.cartesian"][:3]
                controller_error = (
                    obs["observation.state.target"][:3]
                    - obs["observation.state.cartesian"][:3]
                )
            while (
                np.linalg.norm(obs["observation.velocity.cartesian"][:3]) > velocity_err
            ):
                obs, *_ = self._logged_env_step(np.zeros(6))

        return obs

    def yz_i_controller_dyz(self, obs, target):
        err = target[1:3] - obs["observation.state.cartesian"][1:3]
        controller_error = (
            obs["observation.state.target"][1:3]
            - obs["observation.state.cartesian"][1:3]
        )
        if np.any(np.abs(controller_error) > self.i_term_clip):
            return np.zeros(2)
        else:
            return (
                np.clip(
                    err + controller_error,
                    -self.i_term_clip,
                    self.i_term_clip,
                )
                - controller_error
            )

    def x_torque_controller_dx(self, obs):
        y_torque_sensed = np.sum(
            obs["observation.state.sensors_bota_ft_sensor"][3:5]
        ) / np.sqrt(2)
        y_torque_error = self.ft_wrench_target - y_torque_sensed
        x_force_error = y_torque_error / self.env_config.ft_controller_lever_arm
        x_impedance_error = (
            obs["observation.state.target"][0] - obs["observation.state.cartesian"][0]
        )
        if x_force_error > 0 and x_impedance_error < self.x_force_clip:
            dx = -min(
                x_force_error / self.x_force_k, self.x_force_clip - x_impedance_error
            )

        elif x_force_error < 0 and x_impedance_error > -self.x_force_clip:
            dx = -max(
                x_force_error / self.x_force_k, -self.x_force_clip - x_impedance_error
            )
        else:
            dx = 0.0
        # print(
        #     f"[X torque controller] target: {self.ft_wrench_target}, sensed: {y_torque_sensed:.3f}, force_error: {x_force_error:.3f}, impedance_error: {x_impedance_error:.5f}, dx: {dx:.5f}"
        # )
        return dx, x_force_error

    def z_force_controller_dz(self, obs, z_force_target: float = -6.0):
        """Returns (dz, z_force_error). z_force_target follows the FT sensor
        sign convention: negative = EE pressing down on the surface (sensor
        reads the reaction). Default -6 N matches the contact-establishment
        loop. Callers (e.g. snap_push) may pass a larger magnitude for a
        firmer press."""
        z_force_error = (
            z_force_target - obs["observation.state.sensors_bota_ft_sensor"][2]
        )
        z_impedance_error = (
            obs["observation.state.target"][2] - obs["observation.state.cartesian"][2]
        )
        if z_force_error > 0 and z_impedance_error < self.z_force_clip:
            return min(
                z_force_error / self.z_force_k, self.z_force_clip - z_impedance_error
            ), z_force_error

        elif z_force_error < 0 and z_impedance_error > -self.z_force_clip:
            return max(
                z_force_error / self.z_force_k, -self.z_force_clip - z_impedance_error
            ), z_force_error
        else:
            return 0.0, z_force_error

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> tuple[Any, dict[str, Any]]:
        # Start a fresh per-episode FT log. Everything from here until the
        # next reset (or pop_episode_ft_log call) is captured: grasp, PE,
        # contact, policy steps, snap_push.
        self._episode_ft_log = []
        self._ft_logging_active = True
        # Consume the snap-push-homed flag (set when snap_push(reinforce=True)
        # already drove the robot to home). When True the post-rollout
        # sweep AND the trailing home() call are both redundant.
        skip_reset_sweep = self._already_homed_in_snap
        self._already_homed_in_snap = False

        self.env.unwrapped.gripper.set_target(0.85)  # type: ignore
        time.sleep(1.0)

        if not self.first_reset:
            if skip_reset_sweep:
                print(
                    "Skipping reset sweep (snap_push already homed and opened "
                    "the gripper)."
                )
            else:
                # open gripper
                self.env.unwrapped.gripper.set_target(0.85)  # type: ignore
                time.sleep(2.0)
                self.obs, *_ = self._logged_env_step(np.zeros(6))  # wait one step
                # move back and up
                self.obs = self.go_delta(
                    self.obs,
                    [-0.04, 0.0, 0.02],
                )
                # close gripper
                self.env.unwrapped.gripper.set_target(0.5)  # type: ignore
                time.sleep(2.5)

                if self.use_ft_controller:
                    # push with constant force (5N)
                    self.obs, *_ = self._logged_env_step(np.zeros(6))
                    while self.obs["observation.state.sensors_bota_ft_sensor"][2] > -6:
                        if np.linalg.norm(self.obs["observation.velocity.cartesian"]) > 0.0015:
                            self.obs, *_ = self._logged_env_step(np.zeros(6))
                        else:
                            dz, _ = self.z_force_controller_dz(self.obs)
                            self.obs, *_ = self._logged_env_step(
                                np.array([0.0, 0.0, dz, 0.0, 0.0, 0.0])
                            )
                    self._logged_env_step(np.array([0.0, 0.0, -0.02, 0.0, 0.0, 0.0]))
                    time.sleep(2.0)
                else:
                    print("Skipping constant-force push (FT controller disabled).")
        else:
            print("homing first time...")
            # Robust safety lift on first reset (works from gravity-comp).
            # We can't use go_delta here: the cartesian controller is not
            # engaged yet and robot.target_pose is uninitialised, so
            # go_delta's convergence loop hangs forever.
            #
            # Instead, use manipulator_env.move_to(), which is self-contained:
            #   (a) reads the current EE pose from a ROS topic (does NOT
            #       require an active controller),
            #   (b) switches to the cartesian controller,
            #   (c) opens the gripper,
            #   (d) runs a planned trajectory at `speed` m/s,
            #   (e) resets robot.target_pose to the reached pose,
            #   (f) switches back to the default controller.
            self.env.unwrapped.robot.wait_until_ready()  # type: ignore
            current_xyz = np.array(
                self.env.unwrapped.robot.end_effector_pose.position  # type: ignore
            )
            target_xyz = current_xyz + np.array([0.0, 0.0, 0.040])
            print(f"  lift 40 mm:  {current_xyz}  ->  {target_xyz}")
            try:
                self.env.unwrapped.move_to(position=target_xyz, speed=0.03)  # type: ignore
            except Exception as e:
                # Don't trap the user at "homing first time..." if the lift
                # fails (unreachable target, controller refused, etc.) —
                # fall through to the home() call below, which uses the
                # joint trajectory controller and is the original behaviour.
                print(f"  WARNING: move_to lift failed ({e!r}); skipping lift.")
            print("  homing to custom_first_home_position...")
            self.env.unwrapped.home(  # type: ignore
                home_config=self.env_config.custom_first_home_position
            )

        if skip_reset_sweep:
            print("Skipping homing (snap_push already drove robot to home).")
        else:
            # Safety lift before homing, mirroring snap_push reinforce path.
            # Only valid when the cartesian controller is still engaged: NOT
            # on first_reset (the first-time block above ends in a joint-
            # space home() which detaches the cartesian controller, so
            # go_delta would hang).
            if not self.first_reset:
                print(
                    "[InsertionWrapperSiemensPE] reset (no snap) — +Y +Z 30 mm "
                    "diagonal before homing..."
                )
                self.obs = self.go_delta(self.obs, [0.0, 0.030, 0.030])
                print(
                    "[InsertionWrapperSiemensPE] reset (no snap) — waiting "
                    "before homing..."
                )
                self._settle_for(4.0)
            print("homing...")
            self.env.unwrapped.home(home_config=self.home_config)  # type: ignore

        time.sleep(2.0)

        if options is not None and options.get("last_reset", False):
            print("Last reset, not going to start position.")
            return self.obs, {}

        self.obs, reset_info = self.env.reset(seed=seed, options=options)
        self.first_reset = False
        self._pe_episode_idx += 1

        # estimate (coarse)
        _viz_enabled = self._pose_overlay_renderer is not None
        if _viz_enabled:
            world_D_world_obj, _coarse_details = self.estimate_world_pose_once(
                return_details=True
            )
            _coarse_rgb = np.copy(self.obs["observation.images.wrist_camera"])
            _coarse_depth = np.copy(self.obs["observation.images.wrist_depth_camera"])
            _coarse_tcp_cart = np.copy(self.obs["observation.state.cartesian"])
            _coarse_pose_cam = _coarse_details["pose_cam"]
            _coarse_mask = _coarse_details["mask"]
            _coarse_world = np.copy(world_D_world_obj)
        else:
            world_D_world_obj = self.estimate_world_pose_once()
        pose_check_coarse = self.validate_world_pose_transform(
            world_D_world_obj, "coarse"
        )
        est_obj_pos = world_D_world_obj[:3, 3]
        est_obj_euler = rot_matrix_to_euler_xyz(world_D_world_obj[:3, :3])
        print("Estimated object pose (world frame):")
        print(f"  position xyz [m] = {est_obj_pos}")
        print(f"  orientation rpy [rad] = {est_obj_euler}")
        print(f"  orientation rpy [deg] = {np.rad2deg(est_obj_euler)}")
        print("Estimated transform matrix:", world_D_world_obj)
        world_T_demoobj_estiobj = (
            world_D_world_obj[:3, 3] - self.env_config.demo_w_D_w_o[:3, 3]
        )
        print(
            "Delta translation: ",
            world_T_demoobj_estiobj,
        )

        # Apply orientation alignment at pre-grasp hover so the wrist camera
        # approaches refined PE from a demo-aligned viewpoint and the final
        # descent delta is small. 25 deg axis-angle clip in
        # compute_alignment_rpy caps the blast radius if coarse PE is bad.
        # In 3DoF policy mode, compute_alignment_rpy returns yaw-only —
        # applying it just rotates the wrist about world-Z, which is what a
        # top-down grasp on a yawed object needs and is independent of the
        # policy's action dim.
        coarse_alignment_rpy = self.compute_alignment_rpy(world_D_world_obj)
        coarse_rel_euler = coarse_alignment_rpy
        coarse_R_applied = euler_to_rot_matrix(*coarse_rel_euler)
        print(
            "coarse alignment rpy [deg] (applied at hover):",
            np.rad2deg(coarse_rel_euler),
            "(6DoF)" if self._pe_alignment_is_6dof else "(3DoF yaw-only)",
        )
        # Reassigned to the TOTAL applied (coarse + delta) after descent.
        # At viz-dump time below, this reflects what has been applied so far
        # (coarse only).
        applied_alignment_rpy = np.asarray(coarse_rel_euler, dtype=np.float64)

        # Pre-grasp hover: use the rotated grasp offset so a yawed object
        # still ends up centered under the wrist camera for refined PE.
        coarse_grasp_world = (
            world_D_world_obj[:3, 3]
            + world_D_world_obj[:3, :3] @ self.o_T_o_tcpgrasp
        )
        coarse_hover_world = (
            coarse_grasp_world
            + world_D_world_obj[:3, :3] @ np.array([0.0, 0.0, 0.02])
        )
        #self.obs = self.slow_move_to(
        #    coarse_hover_world,
        #     relative_pose_euler=coarse_rel_euler,
        #    max_step=0.0012, 
        #    distance_err=0.0002,
        #    rotation_max_step_rad=np.deg2rad(0.5),
        #)

        # Step 1: move laterally (XY) above the target, keep current Z
        current_z = float(self.obs["observation.state.cartesian"][2])
        xy_above = coarse_hover_world.copy()
        xy_above[2] = current_z

        print("Moving laterally to hover XY (no Z change)...")
        # Use go_to_waypoint for the lateral move + rotation so the
        # controller receives a waypoint target and we wait for convergence.
        self.obs = self.go_to_waypoint(
            self.obs,
            xy_above,
            coarse_rel_euler,
            distance_err=0.010,
            velocity_err=0.010,
            is_via=True,
            coarse_follow_err_tol=0.010,
            coarse_velocity_tol=0.010,
        )
        self._settle_for(0.2)

        # Step 2: descend in Z to the hover height (do NOT re-apply rotation)
        print("Descending to hover Z (vertical only)...")
        # Descend vertically using go_to_waypoint (no rotation re-applied).
        self.obs = self.go_to_waypoint(
            self.obs,
            coarse_hover_world,
            None,  # rotation already applied above
            distance_err=0.010,
            velocity_err=0.010,
            is_via=True,
            coarse_follow_err_tol=0.010,
            coarse_velocity_tol=0.010,
        )
        self._settle_for(0.2)

        # estimate (refined)
        if _viz_enabled:
            world_D_world_obj, _refined_details = self.estimate_world_pose_once(
                return_details=True
            )
            _refined_rgb = np.copy(self.obs["observation.images.wrist_camera"])
            _refined_depth = np.copy(self.obs["observation.images.wrist_depth_camera"])
            _refined_tcp_cart = np.copy(self.obs["observation.state.cartesian"])
            _refined_pose_cam = _refined_details["pose_cam"]
            _refined_mask = _refined_details["mask"]
            _refined_world = np.copy(world_D_world_obj)
        else:
            world_D_world_obj = self.estimate_world_pose_once()
        pose_check_refined = self.validate_world_pose_transform(
            world_D_world_obj, "refined"
        )
        est_obj_pos_refined = world_D_world_obj[:3, 3]
        est_obj_euler_refined = rot_matrix_to_euler_xyz(world_D_world_obj[:3, :3])
        print("Refined estimated object pose (world frame):")
        print(f"  position xyz [m] = {est_obj_pos_refined}")
        print(f"  orientation rpy [rad] = {est_obj_euler_refined}")
        print(f"  orientation rpy [deg] = {np.rad2deg(est_obj_euler_refined)}")

        # Grasp position: use the full est_R so the object-frame grasp offset
        # (o_T_o_tcpgrasp) is placed into the world correctly regardless of
        # object roll/pitch/yaw.
        est_R = world_D_world_obj[:3, :3]
        w_T_w_tcpgrasp = world_D_world_obj[:3, 3] + est_R @ self.o_T_o_tcpgrasp

        # Grasp orientation: preserve "gripper-in-object" relative to demo,
        # target total = est_R_demo @ R_home (see compute_alignment_rpy).
        # The coarse alignment was already applied at the pre-grasp hover,
        # so the refined stage commands only the DELTA from coarse->refined.
        # go_to_waypoint pre-multiplies in world frame:
        #   R_new = R_delta @ (R_coarse @ R_home) == R_refined @ R_home
        #   => R_delta = R_refined @ R_coarse^T
        refined_rel_euler_full = self.compute_alignment_rpy(world_D_world_obj)
        refined_R = euler_to_rot_matrix(*refined_rel_euler_full)
        delta_R = refined_R @ coarse_R_applied.T
        target_rel_euler = rot_matrix_to_euler_xyz(delta_R)
        print(
            "full refined alignment rpy [deg]:",
            np.rad2deg(refined_rel_euler_full),
            " delta applied on descent rpy [deg]:",
            np.rad2deg(target_rel_euler),
            "(6DoF)" if self._pe_alignment_is_6dof else "(3DoF yaw-only)",
        )
        est_Z = est_R[:, 2]
        tilt_cos = float(np.clip(est_Z[2], -1.0, 1.0))
        print(
            "Estimated grasp position (full est_R):",
            w_T_w_tcpgrasp,
            " tilt [deg]:",
            np.rad2deg(np.arccos(tilt_cos)),
        )

        refined_alignment_rpy = refined_rel_euler_full
        print("refined alignment rpy [deg] (total):", np.rad2deg(refined_alignment_rpy))

        if _viz_enabled:
            try:
                from crisp_drl.envs.pose_visualizer import dump_raw_npz

                paths = self._pose_overlay_renderer.save_pair(
                    out_dir=self.pose_viz_dir,
                    episode_idx=self._pe_episode_idx,
                    rgb=_coarse_rgb,
                    mask=_coarse_mask,
                    pose_cam_coarse=_coarse_pose_cam,
                    pose_cam_refined=_refined_pose_cam,
                    rgb_refined=_refined_rgb,
                    mask_refined=_refined_mask,
                )
                raw_path = dump_raw_npz(
                    out_dir=self.pose_viz_dir,
                    episode_idx=self._pe_episode_idx,
                    rgb_coarse=_coarse_rgb,
                    rgb_refined=_refined_rgb,
                    depth_coarse=_coarse_depth,
                    depth_refined=_refined_depth,
                    mask_coarse=_coarse_mask,
                    mask_refined=_refined_mask,
                    pose_cam_coarse=_coarse_pose_cam,
                    pose_cam_refined=_refined_pose_cam,
                    world_pose_coarse=_coarse_world,
                    world_pose_refined=_refined_world,
                    tcp_cartesian_coarse=_coarse_tcp_cart,
                    tcp_cartesian_refined=_refined_tcp_cart,
                    applied_alignment_rpy=np.asarray(applied_alignment_rpy),
                    refined_alignment_rpy=np.asarray(refined_alignment_rpy),
                    pose_check_coarse_det_r=np.array([pose_check_coarse["det_r"]], dtype=np.float32),
                    pose_check_coarse_orth_err=np.array([pose_check_coarse["orth_err"]], dtype=np.float32),
                    pose_check_refined_det_r=np.array([pose_check_refined["det_r"]], dtype=np.float32),
                    pose_check_refined_orth_err=np.array([pose_check_refined["orth_err"]], dtype=np.float32),
                )
                print(f"[pose viz] saved {paths} and {raw_path}")
            except Exception as e:
                print(f"[pose viz] failed to save artifacts: {e}")

        # Final descent: apply target_rel_euler (= coarse->refined DELTA) at
        # the refined hover so gripper-in-object matches the demo. Coarse
        # portion was already commanded before refined PE. Rotation delta
        # is applied ONCE at the hover step and then persists on the robot
        # target_pose; the descent call MUST pass None to avoid doubling
        # the rotation (and breaking the undo at end of this method).
        print("Moving to pre-grasp hover (slow)...")
        # Keep the same hover height, but express it as a single object-frame
        # offset so the nonzero lateral grasp offset is included explicitly.
        hover_xyz = world_D_world_obj[:3, 3] + est_R @ (
            self.o_T_o_tcpgrasp + np.array([0.0, 0.0, 0.015])
        )
        # Slower / more refined than go_to_waypoint: caps controller advance
        # at 0.8 mm/step (~12 mm/s at 15 Hz) so the approach doesn't race.
        self.obs = self.slow_move_to(
            hover_xyz,
            relative_pose_euler=target_rel_euler,
            max_step=0.0015,
            distance_err=0.0005,
        )
        print("Descending to grasp (slow)...")
        # Final descent: 0.5 mm/step (~7.5 mm/s) for a soft, repeatable
        # touchdown on the lid. ~1.5 cm to cover -> ~2 s under nominal control.
        self.obs = self.slow_move_to(
            w_T_w_tcpgrasp,
            relative_pose_euler=None,
            max_step=0.0012,
            distance_err=0.0002,
        )
        # Track the TOTAL applied orientation (coarse @ hover + delta @ descent
        # == refined_rel_euler_full) so the post-pickup "undo" inverses the
        # orientation we actually commanded. 3DoF PE mode didn't rotate ->
        # zeros -> undo is a no-op.
        applied_alignment_rpy = np.asarray(refined_rel_euler_full, dtype=np.float64)
        
        time.sleep(0.2)
        print("Grasping...")
        self.env.unwrapped.gripper.set_target(0.5)  # type: ignore
        time.sleep(2.5)
        self.obs, *_ = self._logged_env_step(np.zeros(6))
        self.actual_grasp_position = np.copy(
            self.obs["observation.state.cartesian"][:3]
        )

        print("Estimating pose at closed gripper...")
        t_D_t_o_at_closed, _grasp_details = (
            self.pose_estimation_helper.estimate_siemens_tcp_frame_coarse(
                self.obs["observation.images.wrist_camera"],
                self.obs["observation.images.wrist_depth_camera"],
                return_details=True,
            )
        )
        print(
            "closed-gripper tcp-frame object translation [m]:",
            t_D_t_o_at_closed[:3, 3],
        )
        if _viz_enabled:
            try:
                grasp_overlay_png = self._pose_overlay_renderer.save_single(
                    out_dir=self.pose_viz_dir,
                    episode_idx=self._pe_episode_idx,
                    rgb=self.obs["observation.images.wrist_camera"],
                    pose_cam_obj=_grasp_details["pose_cam"],
                    mask=_grasp_details["mask"],
                    suffix="grasp_closed",
                    label=f"ep{self._pe_episode_idx} grasp closed",
                )
                print(f"[pose viz] saved grasp closed overlay: {grasp_overlay_png}")
            except Exception as e:
                print(f"[pose viz] failed to save grasp closed overlay: {e}")

        # pick up quickly
        print("Picking up...")
        # slow_move_to instead of go_delta: in 6DoF-alignment mode the wrist
        # is still tilted here (undo happens later, AFTER the TCP-frame PE
        # below, which needs the original camera orientation). A world-Z
        # lift at a tilted pose can hit a joint limit; the rate-limited
        # ramp is gentler on the kinematics and converges with tight
        # tolerance instead of hanging in a 600-step go_delta timeout.
        cur_xyz_pickup = np.copy(self.obs["observation.state.cartesian"][:3])
        rel_xyz = np.asarray(
            self.env_config.relative_motion_after_grasp_pe[:3], dtype=np.float64
        )
        rel_rpy = np.asarray(
            self.env_config.relative_motion_after_grasp_pe[3:], dtype=np.float64
        )
        self.obs = self.slow_move_to(
            cur_xyz_pickup + rel_xyz,
            relative_pose_euler=rel_rpy if np.any(rel_rpy) else None,
            max_step=0.0020,
            rotation_max_step_rad=np.deg2rad(0.8) if np.any(rel_rpy) else None,
            distance_err=0.0005,
            velocity_err=0.0003,
        )

        # estimate grasped delta
        t_D_t_o = self.pose_estimation_helper.estimate_siemens_tcp_frame_coarse(
            self.obs["observation.images.wrist_camera"],
            self.obs["observation.images.wrist_depth_camera"],
        )
        if _viz_enabled:
            try:
                grasped_png = self._pose_overlay_renderer.save_rgb(
                    out_dir=self.pose_viz_dir,
                    episode_idx=self._pe_episode_idx,
                    rgb=self.obs["observation.images.wrist_camera"],
                    suffix="grasped",
                )
                print(f"[pose viz] saved grasped image: {grasped_png}")
            except Exception as e:
                print(f"[pose viz] failed to save grasped image: {e}")
        self.estimated_grasp_delta = (
            t_D_t_o[:3, 3] - self.env_config.demo_t_D_t_o[:3, 3]
        )
        print(f"grasp delta (estimate): {self.estimated_grasp_delta}")

        # compute goal position
        # go to pose estimation position; estimate; compare to demo pose estimation; compute goal position

        self.goal_position = self.goal_position_ground_truth.copy()
        self.goal_position[0] += self.estimated_grasp_delta[0]
        self.goal_position[2] += self.estimated_grasp_delta[2]

        self.start_position = self.goal_position.copy()

        print("Lifting +14 mm in Z before undoing alignment yaw...")
        cur_xyz_pre_undo = np.copy(self.obs["observation.state.cartesian"][:3])
        # Use go_to_waypoint for the lift so the controller sets a waypoint
        # target and we then wait for convergence instead of the rate-limited
        # slow_move_to behavior.
        self.obs = self.go_to_waypoint(
            self.obs,
            cur_xyz_pre_undo + np.array([0.0, 0.0, 0.014]),
            None,
            distance_err=0.0005,
            velocity_err=0.0003,
            is_via=False,
        )
        self._settle_for(0.2)

        applied_alignment_rot = euler_to_rot_matrix(
            applied_alignment_rpy[0],
            applied_alignment_rpy[1],
            applied_alignment_rpy[2],
        )
        undo_alignment_rpy = rot_matrix_to_euler_xyz(applied_alignment_rot.T)
        # Ramp the undo rotation in axis-angle space (correct composition) and
        # WAIT for it to converge before the lift. Sending it as a single env
        # step lets the TCP keep moving while go_delta below computes its
        # target from the in-progress cartesian state -> the lift target ends
        # up off in X/Y by the lever-arm of the residual rotation.
        cur_xyz_for_undo = np.copy(self.obs["observation.state.cartesian"][:3])
        self.obs = self.slow_move_to(
            cur_xyz_for_undo,
            relative_pose_euler=undo_alignment_rpy,
            max_step=0.0012,
            rotation_max_step_rad=np.deg2rad(0.4),
            distance_err=0.0005,
            velocity_err=0.0003,
        )

        print("Yaw undo complete; continuing with post-grasp waypoints...")

        # First post-grasp waypoint: decouple rotation from translation so
        # the wrist re-orients before lateral motion (cleaner motion with a
        # brick in the gripper). Apply rotation in place via slow_move_to
        # (target = current xyz, so only the rotation delta is commanded
        # and the loop just settles), then do the translation at normal
        # speed via go_to_waypoint with no rotation. Subsequent waypoints
        # use go_to_waypoint with combined translation + rotation.
        for i, (pose, res) in enumerate(self.env_config.waypoints_after_grasp):
            target = pose[:3] + self.estimated_grasp_delta
            if i == 0:
                cur_xyz = np.copy(self.obs["observation.state.cartesian"][:3])
                self.obs = self.slow_move_to(
                    cur_xyz,
                    relative_pose_euler=pose[3:],
                    distance_err=res,
                    max_step=0.0020,
                    rotation_max_step_rad=np.deg2rad(0.8),
                )
                self.obs = self.go_to_waypoint(
                    self.obs,
                    target,
                    None,
                    distance_err=res,
                )
            else:
                self.obs = self.go_to_waypoint(
                    self.obs,
                    target,
                    pose[3:],
                    distance_err=res,
                )

        # lower down slowly until z-force is established in steps of 3mm
        if self.use_ft_controller:
            print("Establishing contact...")
            self.obs, *_ = self._logged_env_step(np.zeros(6))
            self.env.tare_ft_sensor(self.obs)  # pyright: ignore[reportAttributeAccessIssue]
            # [s.reset() for s in self.env.unwrapped.sensors]  # pyright: ignore[reportAttributeAccessIssue] # tare ft sensor
            x_step, x_force_error = self.x_torque_controller_dx(self.obs)
            while abs(x_force_error) > 0.1:  # wait until some contact
                # delta_yz = self.yz_i_controller_dyz(self.obs, self.start_position[1:3])
                delta_yz = (
                    np.clip(
                        self.start_position[1:3]
                        - self.obs["observation.state.cartesian"][1:3],
                        -self.i_term_clip,
                        self.i_term_clip,
                    )
                    if np.linalg.norm(self.obs["observation.velocity.cartesian"]) < 0.001
                    else np.zeros(2)
                )
                self.obs, *_ = self._logged_env_step(
                    np.array([x_step, delta_yz[0], delta_yz[1], 0, 0, 0])
                )
                x_step, x_force_error = self.x_torque_controller_dx(self.obs)
        else:
            print("Skipping contact establishment (FT controller disabled).")

        self.n_steps = 0
        self.obs, reset_info = self.env.reset()
        reset_info["reset.grasped.position"] = self.actual_grasp_position
        reset_info["reset.pose_estimation.object_pose_world.position"] = (
            est_obj_pos_refined
        )
        reset_info["reset.pose_estimation.object_pose_world.euler_rpy"] = (
            est_obj_euler_refined
        )
        reset_info["reset.pose_estimation.object_pose_world.matrix"] = world_D_world_obj
        reset_info["reset.pose_estimation.object_pose_world.valid"] = np.array(
            [pose_check_refined["valid"]], dtype=np.float32
        )
        reset_info["reset.pose_estimation.object_pose_world.det_r"] = np.array(
            [pose_check_refined["det_r"]], dtype=np.float32
        )
        reset_info["reset.pose_estimation.object_pose_world.orth_err"] = np.array(
            [pose_check_refined["orth_err"]], dtype=np.float32
        )
        reset_info["reset.pose_estimation.object_pose_world.coarse_valid"] = np.array(
            [pose_check_coarse["valid"]], dtype=np.float32
        )
        reset_info["reset.pose_estimation.alignment_rpy_applied"] = applied_alignment_rpy
        reset_info["reset.pose_estimation.object_pose_tcp_at_closed_gripper.position"] = (
            t_D_t_o_at_closed[:3, 3]
        )
        reset_info["reset.pose_estimation.object_pose_tcp_at_closed_gripper.matrix"] = (
            t_D_t_o_at_closed
        )

        reset_info["reset.grasped.delta_estimated"] = self.estimated_grasp_delta

        print("Reset complete.")
        return self.obs, reset_info

    # ------------------------------------------------------------------ #
    # snap_push (called by SuccessClassificationWrapper on E_SUCCESS_CLS)  #
    # ------------------------------------------------------------------ #
    def _step_translation_clipped(self, dxyz: np.ndarray) -> np.ndarray:
        """Inline 6-dim action helper for the timed reinforce inner loops."""
        action6 = np.array(
            [dxyz[0], dxyz[1], dxyz[2], 0.0, 0.0, 0.0], dtype=np.float64
        )
        out, *_ = self._logged_env_step(action6)
        return out

    def _settle_for(self, seconds: float) -> None:
        """Equivalent of time.sleep(seconds) but keeps stepping zero actions
        so the cartesian controller can converge to its current target.
        Plain time.sleep blocks the Python thread without sending any
        env.step, so the controller hits the next motion block mid-trajectory
        and the robot 'keeps moving' after the sleep."""
        t0 = time.time()
        while time.time() - t0 < seconds:
            self.obs, *_ = self._logged_env_step(np.zeros(6))

    def slow_move_to(
        self,
        target_xyz,
        relative_pose_euler=None,
        max_step: float = 0.0008,
        rotation_max_step_rad: float | None = None,
        max_time: float = 15.0,
        distance_err: float = 0.0003,
        velocity_err: float = 0.0005,
    ):
        """Rate-limited move to an absolute world-frame XYZ.

        Unlike go_to_waypoint, which commands the full delta on the first env
        step and lets the cartesian controller race to the target as fast as
        it can, this ramps the controller's target_pose forward by at most
        ``max_step`` metres per env.step. The robot follows the target at a
        controlled rate, giving a smoother, slower approach — useful for the
        PE pre-grasp hover and descent.

        Rotation:
          * ``rotation_max_step_rad=None`` (default): the full
            ``relative_pose_euler`` is applied on the first step only
            (open-loop, one shot, matches go_to_waypoint convention).
          * ``rotation_max_step_rad=<float>``: rotation is ramped in
            **axis-angle space** — the Euler target is converted to a
            rotvec (axis · θ), split into N chunks each of magnitude
            ≤ rotation_max_step_rad along the SAME axis, then converted
            back to Euler for the action format. Because all chunks share
            an axis, they commute and compose EXACTLY to the original
            rotation (no Euler-decomposition drift, axis is preserved).
        """
        rel = (
            np.asarray(relative_pose_euler, dtype=np.float64)
            if relative_pose_euler is not None
            else np.zeros(3)
        )
        target_xyz = np.asarray(target_xyz, dtype=np.float64)

        # Pre-compute rotation chunks. By splitting along the single
        # axis-angle vector instead of per-Euler-component, the chunks
        # share an axis and compose exactly to R(rel).
        if rotation_max_step_rad is None or not np.any(rel):
            rot_chunks = [rel.copy()] if np.any(rel) else []
        else:
            from scipy.spatial.transform import Rotation as _Rs

            rotvec = _Rs.from_euler("xyz", rel).as_rotvec()
            total_angle = float(np.linalg.norm(rotvec))
            if total_angle <= rotation_max_step_rad:
                rot_chunks = [rel.copy()]
            else:
                n = int(np.ceil(total_angle / rotation_max_step_rad))
                per_step_rotvec = rotvec / n
                per_step_euler = _Rs.from_rotvec(per_step_rotvec).as_euler("xyz")
                rot_chunks = [per_step_euler.copy() for _ in range(n)]

        rot_idx = 0

        def _next_rot_chunk():
            nonlocal rot_idx
            if rot_idx < len(rot_chunks):
                chunk = rot_chunks[rot_idx]
                rot_idx += 1
                return chunk
            return np.zeros(3)

        # First step: send a clipped translation chunk AND a rotation chunk.
        cur = self.obs["observation.state.cartesian"][:3]
        err = target_xyz - cur
        step = np.clip(err, -max_step, max_step)
        self.obs, *_ = self._logged_env_step(
            np.concatenate([step, _next_rot_chunk()])
        )

        t0 = time.time()
        while True:
            cur = self.obs["observation.state.cartesian"][:3]
            err = target_xyz - cur
            vel = float(np.linalg.norm(self.obs["observation.velocity.cartesian"][:3]))
            rot_done = rot_idx >= len(rot_chunks)
            if (
                rot_done
                and np.linalg.norm(err) < distance_err
                and vel < velocity_err
            ):
                break
            if time.time() - t0 > max_time:
                print(
                    f"  [slow_move_to] TIMEOUT after {max_time:.1f}s  "
                    f"err(mm)={np.round(err * 1000, 2)}  "
                    f"|v|(mm/s)={vel * 1000:.2f}  "
                    f"rot_steps_left={len(rot_chunks) - rot_idx}"
                )
                break
            step = np.clip(err, -max_step, max_step)
            self.obs, *_ = self._logged_env_step(
                np.concatenate([step, _next_rot_chunk()])
            )
        return self.obs

    def snap_push(
        self,
        push_distance: float = 0.003,
        pause_before: float = 1.0,
        pause_after: float = 1.0,
        reinforce: bool = False,
        reinforce_lift: float = 0.013,
        reinforce_press: float = 0.012,
        reinforce_post_lift: float = 0.020,
        reinforce_push_offset_x: float = -0.040,
        reinforce_press_force_n: float = -2.85,
        reinforce_press_ramp_timeout_s: float = 2.0,
        reinforce_press_hold_s: float = 2.5,
    ) -> None:
        """Seat the lid after the success classifier fires. Mirrors the
        non-PE InsertionWrapperSiemens.snap_push timings/sequence; the only
        PE-specific change is that the XY anchor is the live PE-derived
        ``self.goal_position`` instead of the GT goal + reset_grasp_delta.

        Press-down is force-controlled (z_force_controller_dz) in two
        phases: (1) ramp down until ``reinforce_press_force_n`` is reached
        (FT-Z target, negative = pressing down), bounded by
        ``reinforce_press_ramp_timeout_s``; (2) hold at that force for
        ``reinforce_press_hold_s`` seconds so the lid actually seats rather
        than just touching and retreating. ``reinforce_press`` is the max
        allowed Z descent from the press starting height — used as a
        safety cap across BOTH phases so the EE cannot drive into the lid
        indefinitely if contact is lost."""

        if not reinforce:
            return

        # ---- reinforce: open / lift / close / press / lift / open / descend / re-grasp ----
        snap_pos = self.obs["observation.state.cartesian"][:3].copy()
        # Anchor XY on the PE-estimated goal (live), matching the non-PE
        # wrapper's "where the lid should be" anchor but using the live PE
        # estimate instead of GT + grasp_delta.
        goal_xy = np.asarray(self.goal_position[:2], dtype=np.float64).copy()
        regrasp_target = np.array(
            [float(goal_xy[0]), float(goal_xy[1]), float(snap_pos[2])]
        )
        xy_shift_mm = (goal_xy - snap_pos[:2]) * 1e3
        print(
            f"[InsertionWrapperSiemensPE] snap_push: reinforce — anchor PE "
            f"goal XY=[{goal_xy[0]*1e3:.1f}, {goal_xy[1]*1e3:.1f}] mm "
            f"(shift from snap = [{xy_shift_mm[0]:+.2f}, {xy_shift_mm[1]:+.2f}] mm), "
            f"snap_z={snap_pos[2]*1e3:.1f} mm"
        )

        # Anchor XY for the press cycle: shifted by reinforce_push_offset_x in
        # X relative to the PE-estimated goal. ALL subsequent Z motions reuse
        # this XY so the gripper never moves laterally over the placed lid
        # while pressing.
        shifted_xy = np.array(
            [float(goal_xy[0]) + reinforce_push_offset_x, float(goal_xy[1])]
        )
        snap_z = float(snap_pos[2])

        print("[InsertionWrapperSiemensPE] snap_push: reinforce — opening gripper...")
        self.env.unwrapped.gripper.set_target(0.85)  # type: ignore
        self._settle_for(1.0)

        # Steps 2 + 3 combined: shift X and lift +Z in one diagonal move to
        # the lift1 target. Subsequent press/lift2 reuse shifted_xy so X
        # stays put through the press cycle.
        print(
            f"[InsertionWrapperSiemensPE] snap_push: reinforce — shifting X by "
            f"{reinforce_push_offset_x * 1e3:+.1f} mm and lifting "
            f"{reinforce_lift*1e3:.1f} mm in Z..."
        )
        lift1_target = np.array([shifted_xy[0], shifted_xy[1], snap_z + reinforce_lift])
        self.obs = self.go_to_waypoint(self.obs, lift1_target)        
        
        print("[InsertionWrapperSiemensPE] snap_push: reinforce — closing gripper...")
        self.env.unwrapped.gripper.set_target(0.60)  # type: ignore
        self._settle_for(1.5)

        # Step 5: press -Z, force-controlled (XY stays at shifted_xy).
        # Targets ``reinforce_press_force_n`` on FT-Z. Stops when (a) the
        # target force is reached, (b) the EE has descended by
        # ``reinforce_press`` from the press starting height (safety cap),
        # or (c) the timeout fires. Velocity gating mirrors the
        # contact-establishment loop to avoid commanding more dz while the
        # arm is still settling.
        if not self.use_ft_controller:
            print(
                "[InsertionWrapperSiemensPE] snap_push: reinforce — FT "
                "controller disabled, skipping press."
            )
        else:
            press_start_z = float(self.obs["observation.state.cartesian"][2])
            press_z_floor = press_start_z - reinforce_press
            print(
                f"[InsertionWrapperSiemensPE] snap_push: reinforce — "
                f"pressing -Z to {reinforce_press_force_n:.1f} N "
                f"(max descent {reinforce_press*1e3:.1f} mm, "
                f"ramp timeout {reinforce_press_ramp_timeout_s:.1f} s, "
                f"hold {reinforce_press_hold_s:.1f} s)..."
            )
            # Tare the FT sensor at the lifted/closed pose so the press
            # target is relative to the no-contact baseline.
            try:
                self.env.tare_ft_sensor(self.obs)  # type: ignore
            except Exception as e:
                print(f"  [press] tare_ft_sensor failed ({e}); continuing.")

            def _press_step():
                """One control tick of the press loop. Velocity gating
                mirrors the contact-establishment loop to avoid windup
                while the arm is still settling."""
                if np.linalg.norm(
                    self.obs["observation.velocity.cartesian"]
                ) > 0.0015:
                    self.obs, *_ = self._logged_env_step(np.zeros(6))
                else:
                    dz, _ = self.z_force_controller_dz(
                        self.obs, z_force_target=reinforce_press_force_n
                    )
                    self.obs, *_ = self._logged_env_step(
                        np.array([0.0, 0.0, dz, 0.0, 0.0, 0.0])
                    )

            # Phase 1: ramp down until target force is reached. Bounded by
            # safety cap and ramp timeout.
            ramp_t0 = time.time()
            reached = False
            safety_hit = False
            while True:
                fz = float(self.obs["observation.state.sensors_bota_ft_sensor"][2])
                cur_z = float(self.obs["observation.state.cartesian"][2])
                if fz <= reinforce_press_force_n:
                    print(
                        f"  [press] phase 1 reached target: fz={fz:.2f} N "
                        f"after {(time.time()-ramp_t0):.2f}s"
                    )
                    reached = True
                    break
                if cur_z <= press_z_floor:
                    print(
                        f"  [press] phase 1 hit max descent: descended "
                        f"{(press_start_z - cur_z)*1e3:.2f} mm, fz={fz:.2f} N"
                    )
                    safety_hit = True
                    break
                if time.time() - ramp_t0 > reinforce_press_ramp_timeout_s:
                    print(
                        f"  [press] phase 1 ramp timeout: fz={fz:.2f} N, "
                        f"descended {(press_start_z - cur_z)*1e3:.2f} mm"
                    )
                    break
                _press_step()

            # Phase 2: hold at force. Keep running the controller so it
            # maintains -reinforce_press_force_n while the lid seats.
            # Safety cap still applies (if the lid sinks past the floor
            # during hold, stop). Skip hold if we already bottomed out.
            if reinforce_press_hold_s > 0.0 and not safety_hit:
                if not reached:
                    print(
                        f"  [press] phase 2 holding (note: target force "
                        f"not reached in phase 1)..."
                    )
                else:
                    print(
                        f"  [press] phase 2 holding "
                        f"{reinforce_press_hold_s:.1f}s at "
                        f"{reinforce_press_force_n:.1f} N..."
                    )
                hold_t0 = time.time()
                while time.time() - hold_t0 < reinforce_press_hold_s:
                    cur_z = float(self.obs["observation.state.cartesian"][2])
                    if cur_z <= press_z_floor:
                        fz = float(
                            self.obs["observation.state.sensors_bota_ft_sensor"][2]
                        )
                        print(
                            f"  [press] phase 2 hit max descent during "
                            f"hold: descended "
                            f"{(press_start_z - cur_z)*1e3:.2f} mm, "
                            f"fz={fz:.2f} N"
                        )
                        break
                    _press_step()
                fz_end = float(
                    self.obs["observation.state.sensors_bota_ft_sensor"][2]
                )
                cur_z_end = float(self.obs["observation.state.cartesian"][2])
                print(
                    f"  [press] hold done: fz={fz_end:.2f} N, total "
                    f"descent={(press_start_z - cur_z_end)*1e3:.2f} mm"
                )

        # Step 6: lift +Z again (XY stays at shifted_xy).
        print(
            f"[InsertionWrapperSiemensPE] snap_push: reinforce — lifting "
            f"{reinforce_post_lift*1e3:.1f} mm in Z..."
        )
        lift2_target = np.array(
            [
                shifted_xy[0],
                shifted_xy[1],
                snap_z + reinforce_lift - reinforce_press + reinforce_post_lift,
            ]
        )
        t0 = time.time()
        while time.time() - t0 < 1.0:
            err = lift2_target - self.obs["observation.state.cartesian"][:3]
            self.obs = self._step_translation_clipped(
                np.clip(err, -self.i_term_clip*2, self.i_term_clip*2)
            )

        print("[InsertionWrapperSiemensPE] snap_push: reinforce — opening gripper...")
        self.env.unwrapped.gripper.set_target(0.85)  # type: ignore
        self._settle_for(1.2)

        print("[InsertionWrapperSiemensPE] snap_push: reinforce — +Z 60 mm diagonal before homing...")
        self.obs = self.go_delta(self.obs, [0.0, 0.000, 0.060])

        print("[InsertionWrapperSiemensPE] snap_push: reinforce — waiting before homing...")
        # self._settle_for(4.0)

        print("[InsertionWrapperSiemensPE] snap_push: reinforce — going home...")
        self.env.unwrapped.home(home_config=self.home_config)  # type: ignore
        self._already_homed_in_snap = True
        print("[InsertionWrapperSiemensPE] snap_push: reinforce done.")

    def step(self, action) -> tuple[Any, Any, bool, bool, dict[str, Any]]:
        # action_input = np.copy(action)
        x_action = (
            self.x_torque_controller_dx(self.obs)[0] if self.use_ft_controller else 0.0
        )
        if self.use_6dof_grasp:
            if action.shape[0] < 5:
                raise ValueError(
                    f"Expected 5D action in 6DoF mode, got shape {action.shape}"
                )
            action = np.array(
                [x_action, action[0], action[1], action[2], action[3], action[4]]
            )
        else:
            action = np.array([x_action, action[0], action[1], 0, 0, 0])

        # apply safety box
        current_pos_yz = self.obs["observation.state.cartesian"][1:3]
        delta_yz = current_pos_yz - self.goal_position[1:3]
        delta_yz_clipped = np.clip(
            delta_yz, -self.safety_box_radius, self.safety_box_radius
        )
        if np.any(delta_yz_clipped != delta_yz):
            correcting_action = delta_yz_clipped - delta_yz
            action[1:3] = correcting_action

        # print(f"[Step] received {action_input}, performing {action}")
        self.obs, reward, terminated, truncated, info = self._logged_env_step(action)

        self.n_steps += 1
        if self.n_steps >= self.step_limit:
            truncated = True

        return self.obs, reward, terminated, truncated, info


class InsertionWrapperSiemensFull(InsertionWrapperSiemensPE):
    """Siemens insertion wrapper for the combined Siemens+Lego layout.

    Behaviourally identical to its parent ``InsertionWrapperSiemensPE``;
    exists as a named class so the contract is explicit at the call site:

      1. Episode starts homed at the OVERVIEW PE pose (the shared viewpoint
         that frames Siemens lid + Lego bricks). Driven by ``self.home_config``
         which is set from ``env_config.custom_home_position_pe`` — the
         orchestrator overrides that field to the combined first-PE joint
         config before constructing the wrapper.
      2. PE estimates the current world-frame lid pose. ``estimated_grasp_delta``
         is derived from ``demo_w_D_w_o`` and that estimate; the actual grasp
         pose = ``grasp_position_ground_truth + estimated_grasp_delta``. So
         the grasp is PE-driven.
      3. After grasping, the wrapper executes ``env_config.waypoints_after_grasp``
         (each waypoint XYZ shifted by ``estimated_grasp_delta``, rotations
         taken as deltas from the captured grasp orientation), then runs the
         FT contact-establishment loop and the policy.
      4. Goal pose comes from ``env_config.goal_position_ground_truth`` plus
         ``estimated_grasp_delta`` (mirrors current Siemens behaviour). For
         the combined layout, leave ``SiemensConfigFull.goal_position_ground_truth``
         inherited from ``SiemensConfig`` (no override) so the goal stays
         fixed.
      5. On success, ``SuccessClassificationWrapper`` triggers
         ``snap_push(reinforce=args.snap_reinforce)``. With ``reinforce=True``
         the snap sequence performs lift / press / re-grasp / lift, then opens
         the gripper, then explicitly homes to ``self.home_config`` (the
         overview PE pose) and sets ``self._already_homed_in_snap=True``.
      6. The next ``reset()`` consumes the flag and skips its post-rollout
         sweep + trailing home. The next episode therefore begins at the
         overview PE pose, completing the loop.

    To use: pair with ``SiemensConfigFull`` (or any SiemensConfig variant)
    where ``custom_home_position_pe`` is the overview pose, and pass
    ``--snap_reinforce`` to the runner so step 5's reinforce branch fires.
    """
    # No behavioural overrides. Adding any here would silently diverge from
    # the parent's reset / snap_push contract — prefer extending the parent
    # or wiring a config flag instead.
    pass

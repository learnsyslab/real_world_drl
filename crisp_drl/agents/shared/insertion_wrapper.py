import signal
import threading
import time
from typing import Any, Dict, Optional
from gymnasium import Wrapper, spaces
import numpy as np

from crisp_drl.agents.shared.insertion_env_config import SiemensConfig
from crisp_drl.envs.pose_estimation_helper import PoseEstimationHelper
from crisp_drl.agents.shared.algorithm_config import Config
from crisp_gym.envs.env_wrapper import LastObservationWrapper

# ---------------------------------------------------------------------------
# Graceful stop support
# ---------------------------------------------------------------------------
# rclpy installs a C-level SIGINT handler that silently suppresses Ctrl+C from
# Python loops.  Callers should invoke install_stop_handler() once the ROS env
# is initialised so that Ctrl+C raises KeyboardInterrupt as expected.

_stop_event = threading.Event()


def _check_stop() -> None:
    if _stop_event.is_set():
        raise KeyboardInterrupt("Stop requested via SIGINT")


def install_stop_handler() -> None:
    """Re-install a Python SIGINT handler after rclpy has overridden it.

    Call this once per process, after the ROS environment has been created.
    Ctrl+C will then raise KeyboardInterrupt from within any movement loop.
    """
    def _handler(sig, frame):
        _stop_event.set()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handler)


class SensorTareWrapper(Wrapper):
    def __init__(
        self,
        env,
        sensor_key: str = "observation.state.sensors_bota_ft_sensor",
        sensor_data_shape=(6,),
    ):
        super().__init__(env)
        self.sensor_key = sensor_key
        self.sensor_offset = np.zeros(sensor_data_shape)

    def step(self, action: Any) -> tuple[Any, Any, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        return self.observation(obs), reward, terminated, truncated, info

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> tuple[Any, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        self.sensor_offset = obs[self.sensor_key]
        return self.observation(obs), info

    def tare_ft_sensor(self, obs):
        # obs[sensor_key] is processed (raw - sensor_offset). Convert to raw by
        # adding back the old offset so that observation() zeros out future readings.
        self.sensor_offset = obs[self.sensor_key] + self.sensor_offset

    def observation(self, obs):
        obs[self.sensor_key] = obs[self.sensor_key] - self.sensor_offset
        return obs


class InsertionWrapper(Wrapper):
    def __init__(
        self,
        env,
        config: Config,
        grasp_randomisation_x_range=(-0.002, 0.002),
        grasp_randomisation_z_range=(0.0005, 0.002),
        goal_position_randomisation_xy_range=(-0.0028, 0.0028),
        safety_box_radius=0.003,
        safety_box_step_size=0.0005,
        step_limit=150,
        minimal_start_goal_distance=0.003,
        is_eval=False,
        use_pose_estimation=False,
        use_ft_controller: bool = True,
    ):
        super().__init__(env)
        self.config = config
        self.home_config = config.custom_home_position
        self.grasp_position_ground_truth = config.grasp_position_ground_truth
        self.goal_position_ground_truth = config.goal_position_ground_truth
        self.grasp_randomisation_x_range = grasp_randomisation_x_range
        self.grasp_randomisation_z_range = grasp_randomisation_z_range
        self.goal_position_randomisation_xy_range = goal_position_randomisation_xy_range
        self.reset_grasp_delta = np.zeros(3)
        self.safety_box_radius = safety_box_radius
        self.safety_box_step_size = safety_box_step_size
        self.step_limit = step_limit
        self.n_since_last_home = 0
        self.first_reset = True
        self.action_space = spaces.Box(-np.inf, np.inf, (2,))
        self.n_steps = 0
        self.minimal_start_goal_distance = minimal_start_goal_distance
        self.is_eval = is_eval
        self.use_pose_estimation = use_pose_estimation
        self.use_ft_controller = use_ft_controller
        self.pose_estimation_helper = (
            PoseEstimationHelper(
                assumed_orientation=config.pose_estimation_assumed_orientation
            )
            if use_pose_estimation
            else None
        )
        self.pose_estimation_position_euler = np.array(
            config.demo_goal_pose_estimation_euler
        )
        print("[InsertionWrapper] [__init__] Eval mode:", is_eval)

        self.z_force_target = -0.7
        self.z_force_k = 2500
        self.z_force_clip = 0.003
        self.i_term_clip = 0.001
        self.reset_lift_height = 0.01  # 0.035
        self.after_grasp_lift_height = 0.016  # 0.05

        self.delta_z_push_reset = 0.003
        self.delta_z_push_reset_step_size = 0.0008
        self.delta_z_push_reset_careful_threshold_distance = 0.003
        self.delta_z_push_reset_careful_threshold_velocity = 0.003

    def go_to_cartesian(
        self, current_obs, target_cartesian=None, delta=None, fine_resolution=None,
        max_settle_iter: int = 30,
    ):
        assert target_cartesian is not None or delta is not None, (
            "Must provide either target_cartesian or delta"
        )
        if target_cartesian is None:
            target_cartesian = current_obs["observation.state.cartesian"][:3] + delta
        obs, *_ = self.env.step(
            target_cartesian - current_obs["observation.state.cartesian"][:3]
        )
        n = 0
        while (
            np.linalg.norm(target_cartesian - obs["observation.state.cartesian"][:3])
            > 0.002
            or np.linalg.norm(obs["observation.velocity.cartesian"][:3]) > 0.001
        ) and n < max_settle_iter:
            _check_stop()
            obs, *_ = self.env.step(np.zeros(3))
            n += 1
        if fine_resolution is not None:
            err = target_cartesian - obs["observation.state.cartesian"][:3]
            n = 0
            while np.linalg.norm(err) > fine_resolution and n < max_settle_iter:
                _check_stop()
                obs, *_ = self.env.step(
                    np.clip(err, -self.i_term_clip, self.i_term_clip)
                )
                err = target_cartesian - obs["observation.state.cartesian"][:3]
                n += 1
            n = 0
            while np.linalg.norm(obs["observation.velocity.cartesian"][:3]) > 0.0005 and n < max_settle_iter:
                _check_stop()
                obs, *_ = self.env.step(np.zeros(3))
                n += 1
        return obs

    def z_force_controller_dz(self, obs):
        z_force_error = (
            self.z_force_target - obs["observation.state.sensors_bota_ft_sensor"][2]
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
        if not self.first_reset:            
            # lift up
            self.obs, *_ = self.env.step(np.zeros(3))  # wait one step
            delta_z = abs(
                self.obs["observation.state.cartesian"][2]
                - self.obs["observation.state.target"][2]
            )
            self.obs = self.go_to_cartesian(
                self.obs,
                delta=np.array([0.0, 0.0, self.reset_lift_height + delta_z]),
            )

            # go back to grasping position
            self.obs = self.go_to_cartesian(
                self.obs,
                target_cartesian=np.array(
                    [
                        self.actual_grasp_position[0],
                        self.actual_grasp_position[1],
                        self.obs["observation.state.cartesian"][2]
                        - self.reset_lift_height,
                    ]
                ),
            )
            # input(
            #     "Went to grasping position for reset. Enter to continue with pushing down..."
            # )

            # push down
            while (
                abs(
                    self.obs["observation.state.cartesian"][2]
                    - self.obs["observation.state.target"][2]
                )
                < self.delta_z_push_reset
            ):
                delta_xy = (
                    self.actual_grasp_position[0:2]
                    - self.obs["observation.state.cartesian"][0:2]
                )
                delta_z = (
                    -self.delta_z_push_reset_step_size
                    if self.obs["observation.velocity.cartesian"][2]
                    > -self.delta_z_push_reset_careful_threshold_velocity
                    or abs(
                        self.actual_grasp_position[2]
                        - self.obs["observation.state.cartesian"][2]
                    )
                    > self.delta_z_push_reset_careful_threshold_distance
                    else 0.0
                )
                self.obs, *_ = self.env.step(
                    np.array([delta_xy[0], delta_xy[1], delta_z])
                )

            delta_z = abs(
                self.obs["observation.state.cartesian"][2]
                - self.obs["observation.state.target"][2]
            )
            self.obs, *_ = self.env.step(np.array([0.0, 0.0, delta_z * 0.8]))
            self.env.unwrapped.gripper.home()  # type: ignore
            time.sleep(0.5)
            self.n_since_last_home += 1
        if self.n_since_last_home >= 4 or self.first_reset:
            print(
                f"self.n_since_last_home={self.n_since_last_home}, first_reset={self.first_reset}, homing..."
            )
            self.env.unwrapped.home(home_config=self.home_config)  # type: ignore
            self.n_since_last_home = 0
            self.first_reset = False

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

        print("Moving to grasp position...")
        self.obs = self.go_to_cartesian(
            self.obs,
            target_cartesian=self.target_grasp_position,
            fine_resolution=0.0002,
        )
        print("Grasping...")
        self.env.unwrapped.gripper.set_target(0.2)  # type: ignore
        time.sleep(1.0)
        self.obs, *_ = self.env.step(np.zeros(3))
        self.actual_grasp_position = np.copy(
            self.obs["observation.state.cartesian"][:3]
        )

        # pick up quickly
        print("Picking up...")
        self.obs = self.go_to_cartesian(
            self.obs, delta=np.array([0.0, 0.0, self.after_grasp_lift_height])
        )

        # compute goal position
        if self.use_pose_estimation and self.pose_estimation_helper is not None:
            # go to pose estimation position; estimate; compare to demo pose estimation; compute goal position
            print("Moving to pose estimation position...")
            self.obs = self.go_to_cartesian(
                self.obs,
                target_cartesian=self.pose_estimation_position_euler[:3],
                fine_resolution=0.0005,
            )
            self.obs, *_ = self.env.step(np.zeros(3))
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
            goal_position_randomisation_xy = np.zeros(2)
            while np.linalg.norm(goal_position_randomisation_xy) < 0.001:
                goal_position_randomisation_xy = np.random.uniform(
                    self.goal_position_randomisation_xy_range[0],
                    self.goal_position_randomisation_xy_range[1],
                    size=2,
                )

            self.goal_position[:2] += goal_position_randomisation_xy
            self.goal_position[0] += (
                self.actual_grasp_position[0] - self.grasp_position_ground_truth[0]
            )
            self.goal_position[2] += (
                self.actual_grasp_position[2] - self.grasp_position_ground_truth[2]
            )

        if not self.is_eval:
            self.start_position = self.goal_position_ground_truth.copy()
            while (
                np.linalg.norm(
                    self.start_position[:2] - self.goal_position_ground_truth[:2]
                )
                < self.minimal_start_goal_distance
            ):
                self.start_position[:2] = self.goal_position[:2] + np.random.uniform(
                    -self.safety_box_radius, self.safety_box_radius, size=2
                )
        else:
            self.start_position = self.goal_position.copy()

        # move to start position quickly
        print("Moving to start position...")
        self.obs = self.go_to_cartesian(
            self.obs,
            target_cartesian=np.array(
                [
                    self.start_position[0],
                    self.start_position[1],
                    self.obs["observation.state.cartesian"][2],
                ]
            ),
            fine_resolution=0.0005,
        )
        self.obs, *_ = self.env.step(
            np.zeros(3)
        )  # wait one step to come to a stop before zeroing ft data
        # lower down slowly until z-force is established in steps of 3mm
        if self.use_ft_controller:
            print("Establishing contact...")
            self.env.tare_ft_sensor(self.obs)  # pyright: ignore[reportAttributeAccessIssue]
            # [s.reset() for s in self.env.unwrapped.sensors]  # pyright: ignore[reportAttributeAccessIssue] # tare ft sensor
            z_step, z_force_error = self.z_force_controller_dz(self.obs)
            while abs(z_force_error) > 0.1:  # wait until some contact
                _check_stop()
                delta_xy = (
                    self.start_position[0:2] - self.obs["observation.state.cartesian"][0:2]
                )
                self.obs, *_ = self.env.step(np.array([delta_xy[0], delta_xy[1], z_step]))
                z_step, z_force_error = self.z_force_controller_dz(self.obs)
        else:
            print("Skipping contact establishment (FT controller disabled).")

        self.n_steps = 0
        self.obs, reset_info = self.env.reset()
        reset_info["reset.grasped.position"] = self.actual_grasp_position
        self.reset_grasp_delta = (
            self.actual_grasp_position - self.grasp_position_ground_truth
        )
        reset_info["reset.grasped.delta_estimated"] = self.reset_grasp_delta
        goal_position_offset = self.goal_position - (
            self.goal_position_ground_truth + self.reset_grasp_delta
        )
        reset_info["reset.goal_position.offset"] = goal_position_offset
        if self.use_pose_estimation:
            reset_info["reset.pose_estimation.lavender"] = lavender_pose  # pyright: ignore[reportPossiblyUnboundVariable]
            reset_info["reset.pose_estimation.purple"] = purple_pose  # pyright: ignore[reportPossiblyUnboundVariable]
            reset_info["reset.pose_estimation.joint_state"] = (
                pose_estimation_joint_state  # pyright: ignore[reportPossiblyUnboundVariable]
            )
            reset_info["reset.pose_estimation.cartesian"] = (
                self.actual_estimation_position
            )  # pyright: ignore[reportPossiblyUnboundVariable]
        print("Reset complete.")
        self.obs = self.add_perfect_action_to_obs(self.obs)
        return self.obs, reset_info

    def step(self, action) -> tuple[Any, Any, bool, bool, dict[str, Any]]:
        z_action = (
            self.z_force_controller_dz(self.obs)[0] if self.use_ft_controller else 0.0
        )
        action = np.array([action[0], action[1], z_action])

        # apply safety box
        current_pos_xy = self.obs["observation.state.cartesian"][:2]
        delta_xy = self.goal_position[:2] - current_pos_xy
        norm_xy = np.linalg.norm(delta_xy)
        if norm_xy > self.safety_box_radius:
            action[:2] = delta_xy * self.safety_box_step_size / norm_xy

        self.obs, reward, terminated, truncated, info = self.env.step(action)
        self.obs = self.add_perfect_action_to_obs(self.obs)

        self.n_steps += 1
        if self.n_steps >= self.step_limit:
            truncated = True

        return self.obs, reward, terminated, truncated, info

    def add_perfect_action_to_obs(self, obs):
        obs["observation.perfect_action"] = obs["observation.state.cartesian"][:3] - (
            self.goal_position_ground_truth + self.reset_grasp_delta
        )
        return obs


class InsertionWrapper3DoFRotZ(Wrapper):
    """Real-world 3-DoF + Rot-Z wrapper.

    Agent action: 3-D ``[dx, dy, drz]``. The wrapper expands to 6-D
    ``[dx, dy, dz_ft, 0, 0, drz]`` for the underlying ``NoGripperActionWrapper``
    (which then pads gripper=0 → 7-D for the ManipulatorCartesianEnv).

    ``observation.state.cartesian[3:6]`` is a rotation vector (axis*angle, rad).
    Home pose has ``rx≈ry≈0`` so ``cartesian[5]`` is pure yaw.
    """

    def __init__(
        self,
        env,
        config: Config,
        grasp_randomisation_x_range=(-0.002, 0.002),
        grasp_randomisation_z_range=(0.0005, 0.002),
        goal_position_randomisation_xy_range=(-0.0028, 0.0028),
        goal_orientation_randomisation_angle=np.deg2rad(3),
        safety_box_radius=0.003,
        safety_box_step_size=0.0005,
        safety_box_angular_radius=np.deg2rad(3),
        safety_box_angular_step_size=np.deg2rad(0.5),
        step_limit=150,
        minimal_start_goal_distance=0.003,
        minimal_start_goal_angle=np.deg2rad(2),
        is_eval=False,
        use_pose_estimation=False,
        use_ft_controller: bool = True,
    ):
        super().__init__(env)
        self.config = config
        self.home_config = config.custom_home_position
        self.grasp_position_ground_truth = config.grasp_position_ground_truth
        self.goal_position_ground_truth = config.goal_position_ground_truth
        self.grasp_randomisation_x_range = grasp_randomisation_x_range
        self.grasp_randomisation_z_range = grasp_randomisation_z_range
        self.goal_position_randomisation_xy_range = goal_position_randomisation_xy_range
        self.goal_orientation_randomisation_angle = goal_orientation_randomisation_angle
        self.reset_grasp_delta = np.zeros(3)
        self.safety_box_radius = safety_box_radius
        self.safety_box_step_size = safety_box_step_size
        self.safety_box_angular_radius = safety_box_angular_radius
        self.safety_box_angular_step_size = safety_box_angular_step_size
        self.step_limit = step_limit
        self.n_since_last_home = 0
        self.first_reset = True
        self.action_space = spaces.Box(-np.inf, np.inf, (3,))
        self.n_steps = 0
        self.minimal_start_goal_distance = minimal_start_goal_distance
        self.minimal_start_goal_angle = minimal_start_goal_angle
        self.is_eval = is_eval
        self.use_pose_estimation = use_pose_estimation
        self.use_ft_controller = use_ft_controller
        self.pose_estimation_helper = (
            PoseEstimationHelper(
                assumed_orientation=config.pose_estimation_assumed_orientation
            )
            if use_pose_estimation
            else None
        )
        self.pose_estimation_position_euler = np.array(
            config.demo_goal_pose_estimation_euler
        )
        print("[InsertionWrapper3DoFRotZ] [__init__] Eval mode:", is_eval)

        self.z_force_target = -0.7
        self.z_force_k = 2500
        self.z_force_clip = 0.003
        self.i_term_clip = 0.001 * 2 # allow more aggressive z action since we have rotation control to help recover from mistakes
        self.reset_lift_height = 0.020
        self.after_grasp_lift_height = 0.016

        self.delta_z_push_reset = 0.003
        self.delta_z_push_reset_step_size = 0.0008
        self.delta_z_push_reset_careful_threshold_distance = 0.003
        self.delta_z_push_reset_careful_threshold_velocity = 0.003

        self.goal_rotation_z = 0.0
        self.start_rotation_z = 0.0

    def _step_zeros(self):
        _check_stop()
        return self.env.step(np.zeros(6))

    def _step_translation(self, dxyz):
        _check_stop()
        action6 = np.array([dxyz[0], dxyz[1], dxyz[2], 0.0, 0.0, 0.0])
        return self.env.step(action6)

    def _step_yaw(self, drz):
        _check_stop()
        action6 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, drz])
        return self.env.step(action6)

    def go_to_cartesian(
        self, current_obs, target_cartesian=None, delta=None, fine_resolution=None,
        max_settle_iter: int = 300,
    ):
        assert target_cartesian is not None or delta is not None, (
            "Must provide either target_cartesian or delta"
        )
        if target_cartesian is None:
            target_cartesian = current_obs["observation.state.cartesian"][:3] + delta
        obs, *_ = self._step_translation(
            target_cartesian - current_obs["observation.state.cartesian"][:3]
        )
        n = 0
        while (
            np.linalg.norm(target_cartesian - obs["observation.state.cartesian"][:3])
            > 0.002
            or np.linalg.norm(obs["observation.velocity.cartesian"][:3]) > 0.001
        ) and n < max_settle_iter:
            obs, *_ = self._step_zeros()
            n += 1
        if n >= max_settle_iter:
            print(
                f"[InsertionWrapper3DoFRotZ] go_to_cartesian: max_settle_iter={max_settle_iter} hit; "
                f"residual={np.linalg.norm(target_cartesian - obs['observation.state.cartesian'][:3])*1e3:.2f} mm"
            )
        if fine_resolution is not None:
            err = target_cartesian - obs["observation.state.cartesian"][:3]
            n = 0
            while np.linalg.norm(err) > fine_resolution and n < max_settle_iter:
                obs, *_ = self._step_translation(
                    np.clip(err, -self.i_term_clip, self.i_term_clip)
                )
                err = target_cartesian - obs["observation.state.cartesian"][:3]
                n += 1
            n = 0
            while np.linalg.norm(obs["observation.velocity.cartesian"][:3]) > 0.0005 and n < max_settle_iter:
                obs, *_ = self._step_zeros()
                n += 1
        return obs

    def _yaw_error_to(self, current_rotvec, target_rz):
        """Robust yaw error via SO(3): goal=[0,0,target_rz] minus current rotvec.

        Returns the z-component of the rotvec rotation error. Robust to small
        rx/ry drift in ``current_rotvec`` (does not assume rx=ry=0).
        """
        goal_rotvec = np.array([0.0, 0.0, float(target_rz)], dtype=np.float64)
        rot_err = LastObservationWrapper._relative_rotation_error(
            goal_rotvec, np.asarray(current_rotvec, dtype=np.float64)
        )
        return float(rot_err[2])

    def go_to_rotation_z(
        self, current_obs, target_rz, tol=np.deg2rad(0.3), max_drive_iter=200, max_settle_iter=30
    ):
        """Drive yaw to ``target_rz``; XY/Z held by impedance controller."""
        obs = current_obs
        rz_err = self._yaw_error_to(obs["observation.state.cartesian"][3:6], target_rz)
        n = 0
        while abs(rz_err) > tol and n < max_drive_iter:
            drz = np.sign(rz_err) * min(
                self.safety_box_angular_step_size, abs(rz_err)
            )
            obs, *_ = self._step_yaw(drz)
            rz_err = self._yaw_error_to(obs["observation.state.cartesian"][3:6], target_rz)
            n += 1
        if n >= max_drive_iter:
            print(
                f"[InsertionWrapper3DoFRotZ] go_to_rotation_z: max_drive_iter "
                f"hit; residual rz_err={np.rad2deg(rz_err):.3f} deg"
            )
        n_settle = 0
        while (
            np.linalg.norm(obs["observation.velocity.angular"]) > np.deg2rad(1.0)
            and n_settle < max_settle_iter
        ):
            obs, *_ = self._step_zeros()
            n_settle += 1
        return obs

    def z_force_controller_dz(self, obs):
        z_force_error = (
            self.z_force_target - obs["observation.state.sensors_bota_ft_sensor"][2]
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
        #time.sleep(1.0)
        if self.first_reset:
            # Initialize inner env (activates Cartesian controller) before any step.
            # Without this, ManipulatorCartesianEnv.switch_to_default_controller()
            # hasn't been called, so _step_zeros() commands are ignored.
            self.obs, _ = self.env.reset()
            self.obs, *_ = self._step_zeros()
            delta_z = abs(
                self.obs["observation.state.cartesian"][2]
                - self.obs["observation.state.target"][2]
            )
            self.obs = self.go_to_cartesian(
                self.obs,
                delta=np.array([0.0, 0.0, self.reset_lift_height + delta_z]),
            )

            self.env.unwrapped.gripper.set_target(0.75)  # type: ignore
            print("Opening gripper...")
            time.sleep(2.0)
        if not self.first_reset:
            # 1) lift FIRST (gripper may still be in contact at episode end —
            #    rotating in contact would shear brick / jam in hole)
            self.obs, *_ = self._step_zeros()
            delta_z = abs(
                self.obs["observation.state.cartesian"][2]
                - self.obs["observation.state.target"][2]
            )
            self.obs = self.go_to_cartesian(
                self.obs,
                delta=np.array([0.0, 0.0, self.reset_lift_height + delta_z]),
            )            

            # 2) rotate yaw back to 0 in air — brick stand expects rz=0 brick
            self.obs = self.go_to_rotation_z(self.obs, target_rz=0.0)

            #time.sleep(2.0)

            # 3) go back to grasping position
            self.obs = self.go_to_cartesian(
                self.obs,
                target_cartesian=np.array(
                    [
                        self.actual_grasp_position[0],
                        self.actual_grasp_position[1],
                        self.obs["observation.state.cartesian"][2]
                        - self.reset_lift_height,
                    ]
                ),
            )

            # push down
            while (
                abs(
                    self.obs["observation.state.cartesian"][2]
                    - self.obs["observation.state.target"][2]
                )
                < self.delta_z_push_reset
            ):
                delta_xy = (
                    self.actual_grasp_position[0:2]
                    - self.obs["observation.state.cartesian"][0:2]
                )
                delta_z = (
                    -self.delta_z_push_reset_step_size
                    if self.obs["observation.velocity.cartesian"][2]
                    > -self.delta_z_push_reset_careful_threshold_velocity
                    or abs(
                        self.actual_grasp_position[2]
                        - self.obs["observation.state.cartesian"][2]
                    )
                    > self.delta_z_push_reset_careful_threshold_distance
                    else 0.0
                )
                self.obs, *_ = self._step_translation(
                    np.array([delta_xy[0], delta_xy[1], delta_z])
                )

            delta_z = abs(
                self.obs["observation.state.cartesian"][2]
                - self.obs["observation.state.target"][2]
            )
            self.obs, *_ = self._step_translation(np.array([0.0, 0.0, delta_z * 0.8]))
            
            #self.env.unwrapped.gripper.set_target(0.5)  # type: ignore
            #print("Opening gripper...")
            #time.sleep(1.0)
            self.env.unwrapped.gripper.home()  # type: ignore
            time.sleep(0.5)
            self.n_since_last_home += 1
        if self.n_since_last_home >= 4 or self.first_reset:
            #self.env.unwrapped.gripper.set_target(0.5)  # type: ignore
            #print("Opening gripper...")
            #time.sleep(1.0)
                
            print(
                f"self.n_since_last_home={self.n_since_last_home}, first_reset={self.first_reset}, homing..."
            )
            self.env.unwrapped.home(home_config=self.home_config)  # type: ignore
            self.n_since_last_home = 0
            self.first_reset = False

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
        
        # self.env.unwrapped.gripper.set_target(0.8)  # type: ignore
        time.sleep(1.0)
        
        print("Moving to grasp position...")
        self.obs = self.go_to_cartesian(
            self.obs,
            target_cartesian=self.target_grasp_position,
            fine_resolution=0.0002,
        )
        print("Grasping...")
        self.env.unwrapped.gripper.set_target(0.5)  # type: ignore
        time.sleep(2.0)
        self.obs, *_ = self._step_zeros()
        self.actual_grasp_position = np.copy(
            self.obs["observation.state.cartesian"][:3]
        )

        # pick up quickly
        print("Picking up...")
        self.obs = self.go_to_cartesian(
            self.obs, delta=np.array([0.0, 0.0, self.after_grasp_lift_height])
        )

        # compute goal position (XY)
        if self.use_pose_estimation and self.pose_estimation_helper is not None:
            print("Moving to pose estimation position...")
            self.obs = self.go_to_cartesian(
                self.obs,
                target_cartesian=self.pose_estimation_position_euler[:3],
                fine_resolution=0.0005,
            )
            self.obs, *_ = self._step_zeros()
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
            goal_position_randomisation_xy = np.zeros(2)
            while np.linalg.norm(goal_position_randomisation_xy) < 0.001:
                goal_position_randomisation_xy = np.random.uniform(
                    self.goal_position_randomisation_xy_range[0],
                    self.goal_position_randomisation_xy_range[1],
                    size=2,
                )
            self.goal_position[:2] += goal_position_randomisation_xy
            self.goal_position[0] += (
                self.actual_grasp_position[0] - self.grasp_position_ground_truth[0]
            )
            self.goal_position[2] += (
                self.actual_grasp_position[2] - self.grasp_position_ground_truth[2]
            )

        # sample goal yaw (absolute world rz; ideal=0)
        goal_position_randomisation_rz = 0.0
        while abs(goal_position_randomisation_rz) < 0.1 * self.minimal_start_goal_angle:
            goal_position_randomisation_rz = np.random.uniform(
                -self.goal_orientation_randomisation_angle,
                self.goal_orientation_randomisation_angle,
            )
        self.goal_rotation_z = goal_position_randomisation_rz

        if not self.is_eval:
            self.start_position = self.goal_position_ground_truth.copy()
            self.start_rotation_z = self.goal_rotation_z
            while (
                np.linalg.norm(
                    self.start_position[:2] - self.goal_position_ground_truth[:2]
                )
                < self.minimal_start_goal_distance
            ):
                self.start_position[:2] = self.goal_position[:2] + np.random.uniform(
                    -self.safety_box_radius, self.safety_box_radius, size=2
                )
            while (
                abs(self.start_rotation_z - self.goal_rotation_z)
                < self.minimal_start_goal_angle
            ):
                self.start_rotation_z = self.goal_rotation_z + np.random.uniform(
                    -self.safety_box_angular_radius, self.safety_box_angular_radius
                )
        else:
            self.start_position = self.goal_position.copy()
            self.start_rotation_z = self.goal_rotation_z

        # move to start XY (rz still 0 from home)
        print("Moving to start position...")
        self.obs = self.go_to_cartesian(
            self.obs,
            target_cartesian=np.array(
                [
                    self.start_position[0],
                    self.start_position[1],
                    self.obs["observation.state.cartesian"][2],
                ]
            ),
            fine_resolution=0.0005,
        )

        # rotate yaw to start_rotation_z BEFORE establishing FT contact
        print(f"Rotating yaw to {np.rad2deg(self.start_rotation_z):.2f} deg...")
        self.obs = self.go_to_rotation_z(self.obs, target_rz=self.start_rotation_z)

        # Do NOT call env.reset() here: switch_to_default_controller() inside
        # ManipulatorCartesianEnv.reset() re-initialises the impedance controller
        # target to a position above hover, making the spring pull the robot UP
        # during contact establishment. The controller is already active and the
        # target is correctly at hover height from the grasp sequence above.
        self.n_steps = 0
        reset_info = {}
        self.obs, *_ = self._step_zeros()
        if self.use_ft_controller:
            print("Establishing contact...")
            self.env.tare_ft_sensor(self.obs)  # pyright: ignore[reportAttributeAccessIssue]
            z_step, z_force_error = self.z_force_controller_dz(self.obs)
            while abs(z_force_error) > 0.1:
                delta_xy = (
                    self.start_position[0:2]
                    - self.obs["observation.state.cartesian"][0:2]
                )
                self.obs, *_ = self._step_translation(
                    np.array([delta_xy[0], delta_xy[1], z_step])
                )
                z_step, z_force_error = self.z_force_controller_dz(self.obs)
        else:
            print("Skipping contact establishment (FT controller disabled).")
        reset_info["reset.grasped.position"] = self.actual_grasp_position
        self.reset_grasp_delta = (
            self.actual_grasp_position - self.grasp_position_ground_truth
        )
        reset_info["reset.grasped.delta_estimated"] = self.reset_grasp_delta
        goal_position_offset = self.goal_position - (
            self.goal_position_ground_truth + self.reset_grasp_delta
        )
        reset_info["reset.goal_position.offset"] = goal_position_offset
        reset_info["reset.goal_orientation.rotation_z"] = self.goal_rotation_z
        reset_info["reset.start_orientation.rotation_z"] = self.start_rotation_z
        if self.use_pose_estimation:
            reset_info["reset.pose_estimation.lavender"] = lavender_pose  # pyright: ignore[reportPossiblyUnboundVariable]
            reset_info["reset.pose_estimation.purple"] = purple_pose  # pyright: ignore[reportPossiblyUnboundVariable]
            reset_info["reset.pose_estimation.joint_state"] = (
                pose_estimation_joint_state  # pyright: ignore[reportPossiblyUnboundVariable]
            )
            reset_info["reset.pose_estimation.cartesian"] = (
                self.actual_estimation_position
            )  # pyright: ignore[reportPossiblyUnboundVariable]
        print("Reset complete.")
        self.obs = self.add_perfect_action_to_obs(self.obs)
        return self.obs, reset_info

    def step(self, action) -> tuple[Any, Any, bool, bool, dict[str, Any]]:
        z_action = (
            self.z_force_controller_dz(self.obs)[0] if self.use_ft_controller else 0.0
        )
        action6 = np.array(
            [action[0], action[1], z_action, 0.0, 0.0, action[2]]
        )

        # XY safety box
        current_pos_xy = self.obs["observation.state.cartesian"][:2]
        delta_xy = self.goal_position[:2] - current_pos_xy
        norm_xy = np.linalg.norm(delta_xy)
        if norm_xy > self.safety_box_radius:
            action6[:2] = delta_xy * self.safety_box_step_size / norm_xy

        # angular safety box (yaw only) — robust SO(3) error
        rz_err = self._yaw_error_to(
            self.obs["observation.state.cartesian"][3:6], self.goal_rotation_z
        )
        if abs(rz_err) > self.safety_box_angular_radius:
            action6[5] = np.sign(rz_err) * self.safety_box_angular_step_size

        self.obs, reward, terminated, truncated, info = self.env.step(action6)
        self.obs = self.add_perfect_action_to_obs(self.obs)

        self.n_steps += 1
        if self.n_steps >= self.step_limit:
            truncated = True

        return self.obs, reward, terminated, truncated, info

    def snap_push(
        self,
        push_distance: float = 0.0030,
        pause_before: float = 1.0,
        pause_after: float = 2.0,
        reinforce: bool = False,
        reinforce_lift: float = 0.0150,
        reinforce_push: float = 0.0200,
        reinforce_post_lift: float = 0.0050,
    ) -> None:
        """Stop → push down push_distance m → hold. Called on E_SUCCESS_CLS.

        If ``reinforce=True``, after the hold phase executes a re-seat cycle:
          open gripper → lift reinforce_lift → close gripper →
          push down reinforce_push → lift reinforce_post_lift →
          open gripper → re-grasp lego → lift reinforce_lift
        """
        print(f"[InsertionWrapper3DoFRotZ] snap_push: pausing {pause_before}s...")
        t0 = time.time()
        while time.time() - t0 < pause_before:
            self.obs, *_ = self._step_zeros()
        print(f"[InsertionWrapper3DoFRotZ] snap_push: pushing down {push_distance*1e3:.1f} mm...")
        self.obs = self.go_to_cartesian(
            self.obs,
            delta=np.array([0.0, 0.0, -push_distance]),
            fine_resolution=push_distance * 0.3,
        )
        print(f"[InsertionWrapper3DoFRotZ] snap_push: holding {pause_after}s...")
        t0 = time.time()
        while time.time() - t0 < pause_after:
            self.obs, *_ = self._step_zeros()
        print("[InsertionWrapper3DoFRotZ] snap_push: done.")

        if not reinforce:
            return

        print("[InsertionWrapper3DoFRotZ] snap_push: reinforce — opening gripper...")
        self.env.unwrapped.gripper.set_target(0.80)  # type: ignore
        time.sleep(1.0)

        print(f"[InsertionWrapper3DoFRotZ] snap_push: reinforce — lifting {reinforce_lift*1e3:.1f} mm...")
        lift1_target = self.obs["observation.state.cartesian"][:3] + np.array([0.0, 0.0, reinforce_lift])
        t0 = time.time()
        while time.time() - t0 < 1.5:
            err = lift1_target - self.obs["observation.state.cartesian"][:3]
            self.obs, *_ = self._step_translation(np.clip(err, -self.i_term_clip, self.i_term_clip))

        print("[InsertionWrapper3DoFRotZ] snap_push: reinforce — closing gripper (press from top)...")
        self.env.unwrapped.gripper.set_target(0.4)  # type: ignore
        time.sleep(1.4)

        print(f"[InsertionWrapper3DoFRotZ] snap_push: reinforce — pressing down {reinforce_push*1e3:.1f} mm...")
        press_target = self.obs["observation.state.cartesian"][:3] + np.array([0.0, 0.0, -reinforce_push])
        t0 = time.time()
        while time.time() - t0 < 1.5:
            err = press_target - self.obs["observation.state.cartesian"][:3]
            self.obs, *_ = self._step_translation(np.clip(err, -self.i_term_clip, self.i_term_clip))

        print(f"[InsertionWrapper3DoFRotZ] snap_push: reinforce — lifting {reinforce_post_lift*1e3:.1f} mm...")
        lift2_target = self.obs["observation.state.cartesian"][:3] + np.array([0.0, 0.0, reinforce_post_lift])
        t0 = time.time()
        while time.time() - t0 < 1.5:
            err = lift2_target - self.obs["observation.state.cartesian"][:3]
            self.obs, *_ = self._step_translation(np.clip(err, -self.i_term_clip, self.i_term_clip))

        print("[InsertionWrapper3DoFRotZ] snap_push: reinforce — opening gripper...")
        self.env.unwrapped.gripper.set_target(0.75)  # type: ignore
        time.sleep(1.0)

        print("[InsertionWrapper3DoFRotZ] snap_push: reinforce — descending to goal position...")
        grasp_target = self.goal_position.copy()
        t0 = time.time()
        while time.time() - t0 < 1.5:
            err = grasp_target - self.obs["observation.state.cartesian"][:3]
            self.obs, *_ = self._step_translation(np.clip(err, -self.i_term_clip, self.i_term_clip))

        print("[InsertionWrapper3DoFRotZ] snap_push: reinforce — re-grasping lego...")
        self.env.unwrapped.gripper.set_target(0.5)  # type: ignore
        time.sleep(1.0)
        print("[InsertionWrapper3DoFRotZ] snap_push: reinforce done.")

    def add_perfect_action_to_obs(self, obs):
        obs["observation.perfect_action"] = obs["observation.state.cartesian"][:3] - (
            self.goal_position_ground_truth + self.reset_grasp_delta
        )
        obs["observation.perfect_rotation"] = -obs["observation.state.cartesian"][3:]
        return obs

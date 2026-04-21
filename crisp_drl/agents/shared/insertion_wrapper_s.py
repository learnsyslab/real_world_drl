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


@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)


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
    ):
        super().__init__(env)
        self.action_space = spaces.Box(-np.inf, np.inf, (2,))
        self.alg_config = alg_config
        self.env_config = env_config
        self.home_config = env_config.custom_home_position
        self.grasp_position_ground_truth = env_config.grasp_position_ground_truth
        self.goal_position_ground_truth = env_config.goal_position_ground_truth
        self.grasp_randomisation_x_range = grasp_randomisation_x_range
        self.grasp_randomisation_z_range = grasp_randomisation_z_range
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
                assumed_orientation=alg_config.pose_estimation_assumed_orientation
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

    def go_to_waypoint(
        self,
        current_obs,
        position,
        relative_pose_euler=None,
        distance_err=0.002,
        velocity_err=0.0005,
        is_via=True,
        is_rotated=False,
    ):
        target = np.concatenate((position, relative_pose_euler or [0.0, 0.0, 0.0]))
        obs, *_ = self.env.step(
            target
            - np.concatenate(
                (current_obs["observation.state.cartesian"][:3], [0.0, 0.0, 0.0])
            )
        )

        # coarse
        while (
            np.any(
                np.abs(
                    obs["observation.state.target"][:3]
                    - obs["observation.state.cartesian"][:3]
                )
                > 0.002
            )
            or np.linalg.norm(obs["observation.velocity.cartesian"][:3]) > 0.001
        ):
            # print(
            #     f"Waiting... (v={np.linalg.norm(obs['observation.velocity.cartesian'][:3])}, err={target[:3] - obs['observation.state.cartesian'][:3]}, controller err={obs['observation.state.target'][:3] - obs['observation.state.cartesian'][:3]})"
            # )
            obs, *_ = self.env.step(np.zeros(6))
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
                    obs, *_ = self.env.step(np.zeros(6))
                else:
                    obs, *_ = self.env.step(
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
                obs, *_ = self.env.step(np.zeros(6))
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
        target = delta + current_obs["observation.state.target"][:3]
        obs, *_ = self.env.step(
            np.concatenate((delta, relative_pose_euler or [0.0, 0.0, 0.0]))
        )

        # coarse
        while (
            np.linalg.norm(target - obs["observation.state.cartesian"][:3]) > 0.002
            or np.linalg.norm(obs["observation.velocity.cartesian"][:3]) > 0.001
        ):
            obs, *_ = self.env.step(np.zeros(6))
        # fine for terminal points
        if not is_via:
            err = target[:3] - obs["observation.state.cartesian"][:3]
            controller_error = (
                obs["observation.state.target"][:3]
                - obs["observation.state.cartesian"][:3]
            )
            while np.linalg.norm(err) > distance_err:
                if np.any(np.abs(controller_error) > self.i_term_clip):
                    obs, *_ = self.env.step(np.zeros(6))
                else:
                    obs, *_ = self.env.step(
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
                obs, *_ = self.env.step(np.zeros(6))

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
        if not self.first_reset:
            # lift up
            self.obs, *_ = self.env.step(np.zeros(6))  # wait one step
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
            # dropoff location
            self.obs = self.go_to_waypoint(
                self.obs,
                self.env_config.dropoff_point,
            )

        else:
            print("homing first time...")
            self.env.unwrapped.home(  # type: ignore
                home_config=self.env_config.custom_first_home_position
            )

            self.first_reset = False
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
        # self.obs, *_ = self.env.step(self.env_config.relative_motion_before_grasp)

        print("Moving to grasp position...")
        self.obs = self.go_to_waypoint(
            self.obs, self.target_grasp_position, distance_err=0.0002, is_via=False
        )
        print("Grasping...")
        self.env.unwrapped.gripper.set_target(0.2)  # type: ignore
        time.sleep(1.0)
        self.obs, *_ = self.env.step(np.zeros(6))
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
            self.env_config.relative_motion_after_grasp[3:],
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
            self.obs, *_ = self.env.step(np.zeros(6))
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
            self.obs, *_ = self.env.step(np.zeros(6))
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
                self.obs, *_ = self.env.step(
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

    def step(self, action) -> tuple[Any, Any, bool, bool, dict[str, Any]]:
        # action_input = np.copy(action)
        x_action = (
            self.x_torque_controller_dx(self.obs)[0] if self.use_ft_controller else 0.0
        )
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
        self.obs, reward, terminated, truncated, info = self.env.step(action)
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
        safety_box_radius=0.002,
        safety_box_step_size=0.0004,
        step_limit=150,
        use_ft_controller: bool = True,
    ):
        super().__init__(env)
        self.use_ft_controller = use_ft_controller
        self.action_space = spaces.Box(-np.inf, np.inf, (2,))
        self.alg_config = alg_config
        self.env_config = env_config
        self.home_config = env_config.custom_home_position_pe
        self.grasp_position_ground_truth = env_config.grasp_position_ground_truth
        self.goal_position_ground_truth = env_config.goal_position_ground_truth
        self.pose_estimation_helper = PoseEstimationHelper(
            assumed_orientation=np.array([])
        )
        self.safety_box_radius = safety_box_radius
        self.safety_box_step_size = safety_box_step_size
        self.step_limit = step_limit
        self.n_since_last_home = 0
        self.first_reset = True
        self.action_space = spaces.Box(-np.inf, np.inf, (2,))
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

        self.o_T_o_tcpgrasp = self.env_config.demo_w_D_w_o[:3, :3].T @ (
            self.grasp_position_ground_truth - self.env_config.demo_w_D_w_o[:3, 3]
        )

    def go_to_waypoint(
        self,
        current_obs,
        position,
        relative_pose_euler=None,
        distance_err=0.002,
        velocity_err=0.0005,
        is_via=True,
        is_rotated=False,
    ):
        target = np.concatenate((position, relative_pose_euler or [0.0, 0.0, 0.0]))
        obs, *_ = self.env.step(
            target
            - np.concatenate(
                (current_obs["observation.state.cartesian"][:3], [0.0, 0.0, 0.0])
            )
        )

        # coarse
        while (
            np.any(
                np.abs(
                    obs["observation.state.target"][:3]
                    - obs["observation.state.cartesian"][:3]
                )
                > 0.002
            )
            or np.linalg.norm(obs["observation.velocity.cartesian"][:3]) > 0.001
        ):
            # print(
            #     f"Waiting... (v={np.linalg.norm(obs['observation.velocity.cartesian'][:3])}, err={target[:3] - obs['observation.state.cartesian'][:3]}, controller err={obs['observation.state.target'][:3] - obs['observation.state.cartesian'][:3]})"
            # )
            obs, *_ = self.env.step(np.zeros(6))
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
                    obs, *_ = self.env.step(np.zeros(6))
                else:
                    obs, *_ = self.env.step(
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
                obs, *_ = self.env.step(np.zeros(6))
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
        target = delta + current_obs["observation.state.target"][:3]
        obs, *_ = self.env.step(
            np.concatenate((delta, relative_pose_euler or [0.0, 0.0, 0.0]))
        )

        # coarse
        while (
            np.linalg.norm(target - obs["observation.state.cartesian"][:3]) > 0.002
            or np.linalg.norm(obs["observation.velocity.cartesian"][:3]) > 0.001
        ):
            obs, *_ = self.env.step(np.zeros(6))
        # fine for terminal points
        if not is_via:
            err = target[:3] - obs["observation.state.cartesian"][:3]
            controller_error = (
                obs["observation.state.target"][:3]
                - obs["observation.state.cartesian"][:3]
            )
            while np.linalg.norm(err) > distance_err:
                if np.any(np.abs(controller_error) > self.i_term_clip):
                    obs, *_ = self.env.step(np.zeros(6))
                else:
                    obs, *_ = self.env.step(
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
                obs, *_ = self.env.step(np.zeros(6))

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

    def z_force_controller_dz(self, obs):
        z_force_target = -6
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
        if not self.first_reset:
            # open gripper
            self.env.unwrapped.gripper.set_target(1.0)  # type: ignore
            time.sleep(2.0)
            self.obs, *_ = self.env.step(np.zeros(6))  # wait one step
            # move back and up
            self.obs = self.go_delta(
                self.obs,
                [-0.04, 0.0, 0.02],
            )
            # close gripper
            self.env.unwrapped.gripper.set_target(0.4)  # type: ignore
            time.sleep(2.0)

            if self.use_ft_controller:
                # push with constant force (5N)
                self.obs, *_ = self.env.step(np.zeros(6))
                while self.obs["observation.state.sensors_bota_ft_sensor"][2] > -6:
                    if np.linalg.norm(self.obs["observation.velocity.cartesian"]) > 0.0015:
                        self.obs, *_ = self.env.step(np.zeros(6))
                    else:
                        dz, _ = self.z_force_controller_dz(self.obs)
                        self.obs, *_ = self.env.step(
                            np.array([0.0, 0.0, dz, 0.0, 0.0, 0.0])
                        )
                self.env.step(np.array([0.0, 0.0, -0.02, 0.0, 0.0, 0.0]))
                time.sleep(2.0)
            else:
                print("Skipping constant-force push (FT controller disabled).")

        print("homing...")
        self.env.unwrapped.home(home_config=self.home_config)  # type: ignore

        time.sleep(3.0)

        if options is not None and options.get("last_reset", False):
            print("Last reset, not going to start position.")
            return self.obs, {}

        self.obs, reset_info = self.env.reset(seed=seed, options=options)
        self.first_reset = False

        # estimate
        world_D_world_obj = (
            self.pose_estimation_helper.estimate_siemens_world_frame_coarse(
                self.obs["observation.images.wrist_camera"],
                self.obs["observation.images.wrist_depth_camera"],
                self.obs["observation.state.cartesian"],
            )
        )
        print("Estimated: ", world_D_world_obj)
        world_T_demoobj_estiobj = (
            world_D_world_obj[:3, 3] - self.env_config.demo_w_D_w_o[:3, 3]
        )
        print(
            "Delta translation: ",
            world_T_demoobj_estiobj,
        )

        world_X_demoobj = self.env_config.demo_w_D_w_o[:3, 0]
        world_X_estiobj = world_D_world_obj[:3, 0]
        relative_rotation_z = np.arctan2(
            np.cross(world_X_demoobj, world_X_estiobj)[2],
            np.dot(world_X_demoobj, world_X_estiobj),
        )

        print(
            "relative rotation around world z-axis: ", np.deg2rad(relative_rotation_z)
        )

        # go to grasp position+20mm adjusted by global delta rotate by +15° and adjust z rotation and y rotation
        self.obs, *_ = self.env.step(
            [0.0, 0.0, 0.0, 0.0, np.deg2rad(-3), relative_rotation_z]
        )
        self.obs = self.go_to_waypoint(
            self.obs,
            self.grasp_position_ground_truth
            + world_T_demoobj_estiobj
            + np.array([0.0, 0.0, 0.02]),
            distance_err=0.0002,
            is_via=False,
        )

        # estimate
        world_D_world_obj = (
            self.pose_estimation_helper.estimate_siemens_world_frame_coarse(
                self.obs["observation.images.wrist_camera"],
                self.obs["observation.images.wrist_depth_camera"],
                self.obs["observation.state.cartesian"],
            )
        )

        w_T_w_tcpgrasp = (
            world_D_world_obj[:3, 3] + world_D_world_obj[:3, :3] @ self.o_T_o_tcpgrasp
        )
        print("Estimated grasp position: ", w_T_w_tcpgrasp)

        world_X_demoobj = self.env_config.demo_w_D_w_o[:3, 0]
        world_X_estiobj = world_D_world_obj[:3, 0]
        relative_rotation_z_v2 = np.arctan2(
            np.cross(world_X_demoobj, world_X_estiobj)[2],
            np.dot(world_X_demoobj, world_X_estiobj),
        )

        print(
            "relative rotation around world z-axis: ",
            np.deg2rad(relative_rotation_z_v2),
        )

        self.obs, *_ = self.env.step(
            [0.0, 0.0, 0.0, 0.0, 0.0, relative_rotation_z_v2 - relative_rotation_z]
        )

        # use delta in gripper frame to demo -> transform into world coordinates and apply
        print("Moving to grasp position...")
        self.obs = self.go_to_waypoint(
            self.obs,
            w_T_w_tcpgrasp,  # + np.array([0.0, 0.0, 0.01]),
            distance_err=0.0002,
            is_via=False,
        )

        print("Grasping...")
        self.env.unwrapped.gripper.set_target(0.2)  # type: ignore
        time.sleep(2.0)
        self.obs, *_ = self.env.step(np.zeros(6))
        self.actual_grasp_position = np.copy(
            self.obs["observation.state.cartesian"][:3]
        )

        # pick up quickly
        print("Picking up...")
        self.obs = self.go_delta(
            self.obs,
            self.env_config.relative_motion_after_grasp_pe[:3],
            self.env_config.relative_motion_after_grasp_pe[3:],
            distance_err=0.001,
            is_via=False,
        )

        # estimate grasped delta
        t_D_t_o = self.pose_estimation_helper.estimate_siemens_tcp_frame_coarse(
            self.obs["observation.images.wrist_camera"],
            self.obs["observation.images.wrist_depth_camera"],
        )
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

        self.obs, *_ = self.env.step([0.0, 0.0, 0.0, 0.0, 0.0, -relative_rotation_z_v2])
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
            self.obs, *_ = self.env.step(np.zeros(6))
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
                self.obs, *_ = self.env.step(
                    np.array([x_step, delta_yz[0], delta_yz[1], 0, 0, 0])
                )
                x_step, x_force_error = self.x_torque_controller_dx(self.obs)
        else:
            print("Skipping contact establishment (FT controller disabled).")

        self.n_steps = 0
        self.obs, reset_info = self.env.reset()
        reset_info["reset.grasped.position"] = self.actual_grasp_position

        reset_info["reset.grasped.delta_estimated"] = self.estimated_grasp_delta

        print("Reset complete.")
        return self.obs, reset_info

    def step(self, action) -> tuple[Any, Any, bool, bool, dict[str, Any]]:
        # action_input = np.copy(action)
        x_action = (
            self.x_torque_controller_dx(self.obs)[0] if self.use_ft_controller else 0.0
        )
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
        self.obs, reward, terminated, truncated, info = self.env.step(action)

        self.n_steps += 1
        if self.n_steps >= self.step_limit:
            truncated = True

        return self.obs, reward, terminated, truncated, info

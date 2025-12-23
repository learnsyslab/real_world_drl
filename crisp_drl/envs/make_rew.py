from crisp_drl.agents.shared.config import Config
from crisp_drl.agents.shared.rewards import (
    prune_after_async_termination,
    sparse_event_reward,
)


def create_sim_reward_fn(
    config: Config,
    max_rew=None,
    ideal_goal_pos_xy=None,
    ideal_grasp_pos_xy=None,
    event_reward_map=None,
):
    # all_rewards = xy_dense_simple_place_reward(
    #     all_actions,
    #     all_observations,
    #     all_rewards,
    #     all_infos,
    #     max_rew=0.1,
    #     gamma=config.gamma,
    #     ideal_goal_pos_xy=np.array([0.6, 0.0]),
    #     ideal_grasp_pos_xy=np.array([0.0, 0.0]),
    #     actual_grasp_pos_xy=actual_grasp_pos[:2],
    #     event_reward_map={
    #         "E_SUCCESS": 0.1 / (1 - config.gamma) * 3,
    #         "E_FAIL": -0.1 / (1 - config.gamma) / 2 * 3,
    #         "E_SAFETY_BOX_VIOLATION": 0.0,  # -0.05,
    #     },
    # )
    # all_rewards = xy_dense_delta_place_reward(
    #     all_actions,
    #     all_observations,
    #     all_rewards,
    #     all_infos,
    #     ideal_goal_pos_xy=goal_pos,
    #     ideal_grasp_pos_xy=ideal_grasp_pos,
    #     actual_grasp_pos_xy=actual_grasp_pos,
    # )
    # all_rewards = xy_action_magnitude_dense_reward(
    #     all_rewards,
    #     all_actions,
    # )
    # all_rewards = dense_place_reward(all_actions, all_observations, all_infos, {"E_TORQUE": -10.0, "E_FAR_AWAY": -3.0, "E_BELOW_Z": -3.0, "E_SUCCESS": 10.0, "E_FAIL": -1.0, "E_BAD_BEHAVIOR": -5.0, "E_CONTROLLER_ISSUE": 0.0},
    #              max_rew=0.01, ideal_goal_pos=[0.53975, -0.033,  0.05], ideal_grasp_pos=np.array([0.58833, -0.13817,  0.04229]), actual_grasp_pos=actual_grasp_pos, k_xy=0.005, k_z=0.002)
    # all_rewards = sparse_place_reward(
    #     all_actions, all_observations, all_infos
    # )
    # all_rewards = xy_dense_place_reward(all_actions, all_observations, all_infos, max_rew=0.01, max_action_magnitude=0.0008,
    #                                      ideal_goal_pos_xy=np.array([0.53975, -0.033]), ideal_grasp_pos_xy=np.array([0.58833, -0.13817]),
    #                                      actual_grasp_pos_xy=actual_grasp_pos[:2])
    # all_rewards = xy_dense_simple_place_reward(
    #     all_actions,
    #     all_observations,
    #     all_rewards,
    #     all_infos,
    #     max_rew=0.1,
    #     gamma=self.config.gamma,
    #     ideal_goal_pos_xy=np.array([0.6, 0.0]),
    #     ideal_grasp_pos_xy=np.array([0.0, 0.0]),
    #     actual_grasp_pos_xy=actual_grasp_pos[:2],
    # )
    # all_rewards = sparse_event_reward(
    #     all_actions,
    #     all_observations,
    #     all_rewards,
    #     all_infos,
    #     {
    #         "E_SUCCESS": 0.1 / (1 - self.config.gamma) / 2 * 3,
    #         "E_FAIL": -0.1 / (1 - self.config.gamma) / 2 * 3,
    #         "E_SAFETY_BOX_VIOLATION": -0.05,
    #     },
    # )
    # all_rewards = xy_action_magnitude_dense_reward(
    #     all_rewards,
    #     all_actions,
    #     threshold=0.00026,
    #     reward=-0.05,
    # )
    def reward_fn(
        all_actions,
        all_observations,
        all_rewards,
        all_infos,
        actual_grasp_pos_xy=None,
    ):
        all_actions, all_observations, all_rewards, all_infos = (
            prune_after_async_termination(
                all_actions,
                all_observations,
                all_rewards,
                all_infos,
                {"E_CONTROLLER_ISSUE", "E_TORQUE"},
            )
        )

        all_rewards = sparse_event_reward(
            all_actions,
            all_observations,
            all_rewards,
            all_infos,
            event_reward_map,
        )
        return all_actions, all_observations, all_rewards, all_infos

    return reward_fn

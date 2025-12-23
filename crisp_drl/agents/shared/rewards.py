import numpy as np
import torch

EVENT_REWARD_MAP = {
    "E_TORQUE": -10.0,
    "E_FAR_AWAY": -3.0,
    "E_BELOW_Z": -3.0,
    "E_SUCCESS": 10.0,
    "E_FAIL": -2.0,
    "E_BAD_BEHAVIOR": -5.0,
    "E_CONTROLLER_ISSUE": 0.0,
}


def sparse_event_reward(
    action_sequence, observation_sequence, reward_sequence, infos, event_reward_map
):
    action_timestamps = list(map(lambda info: info["action_t"], infos[1:]))
    for info in infos[1:]:
        if "custom_events" not in info:
            continue
        timestamp, event = info["custom_events"][0]
        if len(info["custom_events"]) > 1:
            earliest = min(info["custom_events"], key=lambda x: x[0])
            print(
                f"[REWARD] WARNING: More than one custom event: {info['custom_events']}, using earliest: {earliest}"
            )
            timestamp, event = earliest

        i_event = len(action_sequence) - 1
        for i, action_t in enumerate(action_timestamps):
            if action_t > timestamp:
                i_event = i - 1

        reward_sequence[i_event] += event_reward_map[event]
    return reward_sequence


def prune_after_async_termination(
    action_sequence: list,
    observation_sequence: list,
    reward_sequence: list,
    infos: list,
    async_terminating_events: set[str],
) -> tuple:
    # find the earliest ending event, add it to the correct info dict and prune everything after
    timestamp = None
    event = None
    for info in infos:
        if "custom_events" not in info:
            continue
        events = info["custom_events"]
        for timestamp_, event_ in events:
            if event in async_terminating_events:
                if timestamp is None or timestamp_ < timestamp:
                    timestamp = timestamp_
                    event = event_

    if timestamp is None or infos[-1]["action_t"] < timestamp:
        return action_sequence, observation_sequence, reward_sequence, infos

    for i, info in enumerate(infos):
        if info["action_t"] > timestamp:
            if event not in list(map(lambda xy: xy[1], infos[i - 1]["custom_events"])):
                infos[i - 1]["custom_events"].insert(0, (timestamp, event))
            return (
                action_sequence[:i],
                observation_sequence[:i],
                reward_sequence[:i],
                infos[:i],
            )

    assert False, "unreachable"


def dense_place_reward(
    action_sequence,
    observation_sequence,
    infos,
    event_reward_map=EVENT_REWARD_MAP,
    max_rew=0.01,
    k_xy: float = None,
    k_z: float = None,
    ideal_goal_pos=np.array([0.53975, -0.033, 0.05]),
    ideal_grasp_pos=np.array([0.58833, -0.13817, 0.04229]),
    actual_grasp_pos: np.ndarray = None,
):
    # w_motion = 1.0
    rewards = []
    estimated_goal_pos = ideal_goal_pos
    estimated_goal_pos[0] += actual_grasp_pos[0] - ideal_grasp_pos[0]
    estimated_goal_pos[2] += actual_grasp_pos[2] - ideal_grasp_pos[2]
    for i, action, observation in zip(
        range(1000000), action_sequence, observation_sequence
    ):
        rew = 0.0
        # last_i = max(0, i-1)
        # rew += w_motion * np.linalg.norm(observation_sequence[i]["observation.state.cartesian"][:3] - observation_sequence[last_i]["observation.state.cartesian"][:3])
        delta_xy = np.linalg.norm(
            infos[i + 1]["observation"]["observation.state.cartesian"][:2]
            - estimated_goal_pos[:2]
        )
        delta_z = np.abs(
            infos[i + 1]["observation"]["observation.state.cartesian"][2]
            - estimated_goal_pos[2]
        )
        rew += max_rew / 2 * (np.exp(-delta_xy / k_xy) + np.exp(-delta_z / k_z))
        rewards.append(rew)
    return _add_sparse_reward(
        action_sequence, observation_sequence, infos, event_reward_map, rewards
    )


def xy_dense_delta_place_reward(
    action_sequence,
    observation_sequence,
    rewards,
    infos,
    event_reward_map=EVENT_REWARD_MAP,
    max_rew=0.01,
    max_action_magnitude=0.0008,
    ideal_goal_pos_xy=np.array([0.53975, -0.033]),
    ideal_grasp_pos_xy=np.array([0.58833, -0.13817]),
    actual_grasp_pos_xy: np.ndarray = None,
):
    estimated_goal_pos_xy = ideal_goal_pos_xy
    estimated_goal_pos_xy[0] += actual_grasp_pos_xy[0] - ideal_grasp_pos_xy[0]
    for i, action, observation in zip(
        range(1000000), action_sequence, observation_sequence
    ):
        rew = 0.0

        delta_xy = (
            infos[i + 1]["observation"]["observation.state.cartesian"][:2]
            - estimated_goal_pos_xy
        )
        action_xy = action[:2]
        action_xy_angle = np.arctan2(action_xy[1], action_xy[0])
        delta_xy_angle = np.arctan2(delta_xy[1], delta_xy[0])
        correct_magnitude = min(max_action_magnitude, np.linalg.norm(delta_xy))
        angle_diff = np.abs(delta_xy_angle - action_xy_angle)
        angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
        rew_angle = 1 - 2 * angle_diff / np.pi  # normalized to [-1, 1]
        rew_magnitude = (
            1
            - abs(np.linalg.norm(action_xy) - correct_magnitude) / max_action_magnitude
        )

        rew += rew_angle * rew_magnitude * max_rew
        rewards[i] += rew
    return rewards


def xy_action_magnitude_dense_reward(
    rewards,
    action_sequence,
    threshold=0.000251,
    reward=-0.05,
):
    for i, action in zip(range(1000000), action_sequence):
        if type(action) is torch.Tensor:
            action_magnitude = torch.norm(action[:2])
        elif type(action) is np.ndarray:
            action_magnitude = np.linalg.norm(action[:2])
        else:
            raise ValueError(f"Unsupported action type: {type(action)}")
        if action_magnitude > threshold:
            rewards[i] += reward
    return rewards


def xy_dense_simple_place_reward(
    action_sequence,
    observation_sequence,
    reward_sequence,
    infos,
    event_reward_map=EVENT_REWARD_MAP,
    max_rew=0.01,
    sigma=1.0,
    gamma=0.97,
    ideal_goal_pos_xy=np.array([0.53975, -0.033]),
    ideal_grasp_pos_xy=np.array([0.58833, -0.13817]),
    actual_grasp_pos_xy: np.ndarray = None,
):
    # if event_reward_map["E_SUCCESS"] != max_rew / (1 - gamma) / 2:
    #     print(
    #         f"[REWARD] WARNING: E_SUCCESS reward {event_reward_map['E_SUCCESS']} does not match dense reward max_rew "
    #         f"{max_rew} and gamma {gamma}"
    #     )
    # if event_reward_map["E_FAIL"] != -max_rew / (1 - gamma) / 2:
    #     print(
    #         f"[REWARD] WARNING: E_FAIL reward {event_reward_map['E_FAIL']} does not match dense reward max_rew "
    #         f"{max_rew} and gamma {gamma}"
    #     )

    estimated_goal_pos_xy = ideal_goal_pos_xy.copy()
    estimated_goal_pos_xy[0] += actual_grasp_pos_xy[0] - ideal_grasp_pos_xy[0]
    for i, action, observation in zip(
        range(1000000), action_sequence, observation_sequence
    ):
        rew = 0.0

        delta_xy_mm = (
            infos[i + 1]["observation"]["observation.state.cartesian"][:2]
            - estimated_goal_pos_xy
        ) * 1000.0  # in mm

        rew += (
            np.exp(-np.dot(delta_xy_mm, delta_xy_mm) / (2 * sigma**2)) * max_rew
            # np.exp(-np.linalg.norm(delta_xy_mm) / sigma) * max_rew
            - max_rew / 2
        )
        reward_sequence[i] += rew
    return reward_sequence

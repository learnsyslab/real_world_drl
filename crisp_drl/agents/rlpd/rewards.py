import numpy as np

# sparse: successful completion, unsuccessful completion, torque limits, far away
# dense: smoothness: small negative reward for real movement
def place_reward(action_sequence, observation_sequence, infos, event_reward_map):
    # w_motion = 1.0
    rewards = []
    for i, action, observation in zip(range(1000000), action_sequence, observation_sequence):
        rew = 0.0
        # last_i = max(0, i-1)
        # rew += w_motion * np.linalg.norm(observation_sequence[i]["observation.state.cartesian"][:3] - observation_sequence[last_i]["observation.state.cartesian"][:3]) 
        rewards.append(rew)
    action_timestamps = list(map(lambda info: info["action_t"], infos))
    for info in infos:
        if "custom_events" not in info:
            continue
        timestamp, event = info["custom_events"][0]
        if len(info["custom_events"]) > 1:
            earliest = min(info["custom_events"], key=lambda x: x[0])
            print(f"[REWARD] WARNING: More than one custom event: {info['custom_events']}, using earliest: {earliest}")
            timestamp, event = earliest


        i_event = len(action_sequence) - 1
        for i, action_t in enumerate(action_timestamps):
            if action_t > timestamp:
                i_event = i-1

        rewards[i_event] += event_reward_map[event]

    return rewards


def dense_place_reward(action_sequence, observation_sequence, infos, event_reward_map, max_rew=0.01, k_xy=None, k_z=None, ideal_goal_pos=np.array([0.53975, -0.033,  0.05]), ideal_grasp_pos = np.array([0.58833, -0.13817,  0.04229]), actual_grasp_pos = None):
    # w_motion = 1.0
    rewards = []
    estimated_goal_pos = ideal_goal_pos
    estimated_goal_pos[0] += actual_grasp_pos[0] - ideal_grasp_pos[0] 
    estimated_goal_pos[2] += actual_grasp_pos[2] - ideal_grasp_pos[2] 
    for i, action, observation in zip(range(1000000), action_sequence, observation_sequence):
        rew = 0.0
        # last_i = max(0, i-1)
        # rew += w_motion * np.linalg.norm(observation_sequence[i]["observation.state.cartesian"][:3] - observation_sequence[last_i]["observation.state.cartesian"][:3]) 
        delta_xy = np.linalg.norm(infos[i]["observation"]["observation.state.cartesian"][:2] - estimated_goal_pos[:2])
        delta_z = np.abs(infos[i]["observation"]["observation.state.cartesian"][2] - estimated_goal_pos[2])
        rew += max_rew / 2 * (np.exp(-delta_xy / k_xy) + np.exp(-delta_z / k_z))
        rewards.append(rew)
    action_timestamps = list(map(lambda info: info["action_t"], infos))
    for info in infos:
        if "custom_events" not in info:
            continue
        timestamp, event = info["custom_events"][0]
        if len(info["custom_events"]) > 1:
            earliest = min(info["custom_events"], key=lambda x: x[0])
            print(f"[REWARD] WARNING: More than one custom event: {info['custom_events']}, using earliest: {earliest}")
            timestamp, event = earliest

        i_event = len(action_sequence) - 1
        for i, action_t in enumerate(action_timestamps):
            if action_t > timestamp:
                i_event = i-1

        rewards[i_event] += event_reward_map[event]

    return rewards
    
            


def prune_after_async_termination(action_sequence: list, observation_sequence: list, infos: list, terminating_events: set[str]):
    # find the earliest ending event, add it to the correct info dict and prune everything after
    timestamp = None
    event = None
    for info in infos:
        if "custom_events" not in info:
            continue
        events = info["custom_events"]
        for timestamp_, event_ in events:
            if event in terminating_events:
                if timestamp is None or timestamp_ < timestamp:
                    timestamp = timestamp_
                    event = event_

    if timestamp is None or infos[-1]["action_t"] < timestamp:
        return action_sequence, observation_sequence, infos
    
    for i, info in enumerate(infos):
        if info["action_t"] > timestamp:
            if event not in list(map(lambda xy: xy[1], infos[i-1]["custom_events"])):
                infos[i-1]["custom_events"].insert(0, (timestamp, event))
            return action_sequence[:i], observation_sequence[:i], infos[:i]

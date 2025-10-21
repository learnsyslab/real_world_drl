import numpy as np

# sparse: successful completion, unsuccessful completion, torque limits, far away
# dense: smoothness: small negative reward for real movement
def place_reward(action_sequence, observation_sequence, infos, event_reward_map):
    # w_motion = 1.0
    rewards = []
    for i, action, observation in zip(range(1000000), action_sequence, observation_sequence):
        rew = 0.0
        # last_i = max(0, i-1)
        # rew += w_motion * np.linalg.norm(observation[i]["observation.state.cartesian"] - observation[last_i]["observation.state.cartesian"]) 
        rewards.append(rew)
    action_timestamps = list(map(lambda info: info["action_t"], infos))
    for info in infos:
        if "custom_events" not in info:
            continue
        timestamp, event = info["custom_events"]

        i_event = len(action_sequence) - 1
        for i, action_t in enumerate(action_timestamps):
            if action_t > timestamp:
                i_event = i-1

        rew[i_event] += event_reward_map[event]

    return rewards
    
            


def prune_after_async_termination(action_sequence: list, observation_sequence: list, infos: list):
    timestamp = None
    for info in infos:
        if "custom_events" not in info:
            continue
        events = info["custom_events"]
        for timestamp_, event in events:
            if event in ["E_CONTROLLER_ISSUE", "E_TORQUE"]:
                if not timestamp:
                    timestamp = timestamp_
                else:
                    timestamp = min(timestamp_, timestamp)

    if not timestamp:
        return
    
    for i, info in enumerate(infos):
        if info["action_t"] > timestamp:
            return action_sequence[:i], observation_sequence[:i], infos[:i]
    

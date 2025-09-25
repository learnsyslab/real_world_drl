import torch
import numpy as np
from collections import defaultdict
import time
from torchvision.models import ResNet18_Weights


RESNET18_TRANSFORM = ResNet18_Weights.DEFAULT.transforms(antialias=True)


def crisp_obs_to_tensor(observation: dict,  
                        image_encoders: list[torch.nn.Module],
                        device: torch.device,
                        copy: bool=True) -> torch.Tensor:
    """
    Convert an observation dictionary to a tensor.
    """
    tensor_list = []
    camera_counter = 0
    for key, value in observation.items():
        if 'image' in key:
            if len(value.shape) == 3:
                value = np.expand_dims(value, axis=0)  # add batch dimension
            tensor_list.append(encode_image(value, image_encoders[camera_counter], device))
            camera_counter += 1
        elif copy:
            if len(value.shape) == 1:
                value = np.expand_dims(value, axis=0)  # add batch dimension
            tensor_list.append(torch.tensor(value, dtype=torch.float32).to(device))
        else:
            if len(value.shape) == 1:
                value = np.expand_dims(value, axis=0)  # add batch dimension
            tensor_list.append(torch.as_tensor(value, dtype=torch.float32).to(device))
    return torch.cat(tensor_list, dim=-1)


def crisp_batch_obs_to_tensor(observations: list[dict], image_encoders: list[torch.nn.Module], device: torch.device, copy: bool=True) -> torch.Tensor:
    """
    Convert a batch of observation dictionaries to a batched tensor.
    """
    merged = defaultdict(list)
    for obs in observations:
        for key, value in obs[0].items():
            merged[key].append(value)
    batched_observations = {key: np.stack(value) for key, value in merged.items()}
    return crisp_obs_to_tensor(batched_observations, image_encoders, device, copy)


def get_input_size_from_dict_space(obs_space) -> int:
    """
    Get the input size from the observation space.

    :param obs_space: The observation space
    :return: The input size
    """
    size = 0
    for name, space in obs_space.items():
        if 'image' in name:
            size += 10
        else:
            size += np.prod(space.shape)
    return size


def encode_image(img: np.ndarray, image_encoder: torch.nn.Module, device: torch.device) -> torch.Tensor:
    """Encode an image using a ResNet18 model with a custom projection head."""
    img_tensor = torch.from_numpy(img).float() / 255.0
    if img.shape[1] != 3:
        img_tensor = img_tensor.permute(0, 3, 1, 2)
    # preprocess the image for resnet 18
    img_tensor = RESNET18_TRANSFORM(img_tensor)
    # get the features from the image encoder
    features = image_encoder(img_tensor.to(device))
    return features


def load_buffer_from_lerobot_dataset(dataset, buffer, num_episodes: int = None):
    """
    Load a dataset from the LeRobotDataset into a replay buffer.

    :param dataset: The dataset to load from
    :param buffer: The replay buffer to load into
    :param num_episodes: The number of episodes to load (default: None = load all)
    """
    total_recorded_steps = len(dataset)

    current_obs = {}
    for key in dataset[0].keys():
        if 'observation' in key:
            current_obs.update({key: dataset[0][key].numpy()})
    next_obs = {}
    reward = np.zeros(1, dtype=np.float32)
    done = np.zeros(1, dtype=bool)



    for idx in range(total_recorded_steps):
        # Stop if we have loaded the desired number of episodes
        if num_episodes is not None and dataset[idx]['episode_index'] >= num_episodes:
            break
        start_time = time.time()
        # load the next observation
        next_obs = {}
        for key in dataset[idx].keys():
            if 'observation' in key:
                next_obs.update({key: dataset[idx][key].numpy()})
        
        # check if the episode is done
        if idx > 0:
            if dataset[idx]['episode_index'] != dataset[idx - 1]['episode_index']:
                done = np.ones(1, dtype=bool)

        action = dataset[idx]['action'].numpy()
        
        # add information to the buffer
        buffer.add(obs=current_obs,
                   next_obs=next_obs,
                   action=action,
                   reward=reward,
                   done=done,
                   infos={})
        end_time = time.time()
        print(f"total time taken for step {idx}: {end_time - start_time:.4f} seconds")

        # update the current observation
        current_obs = next_obs
        # reset done
        done = np.zeros(1, dtype=bool)
        

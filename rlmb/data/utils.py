import torch
import numpy as np
from collections import defaultdict


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
            size += 128
        else:
            size += np.prod(space.shape)
    return size


def encode_image(img: np.ndarray, image_encoder: torch.nn.Module, device: torch.device) -> torch.Tensor:
    """Encode an image using a ResNet18 model with a custom projection head."""
    img_tensor = torch.as_tensor(img, dtype=torch.float32).permute(0, 3, 1, 2)
    features = image_encoder(img_tensor.to(device))
    return features

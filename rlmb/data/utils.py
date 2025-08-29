import torch
import numpy as np


def crisp_obs_to_tensor(observation: dict, copy: bool=True) -> torch.Tensor:
    """
    Convert a single observation dictionary to a tensor.
    """
    tensor_list = []
    for value in observation.values():
        if copy:
            tensor_list.append(torch.tensor(value, dtype=torch.float32).view(1,-1))
        else:
            tensor_list.append(torch.as_tensor(value, dtype=torch.float32).view(1,-1))
    return torch.cat(tensor_list, dim=-1)


def crisp_batch_obs_to_tensor(observations: list[dict], copy: bool=True) -> torch.Tensor:
    """
    Convert a batch of observation dictionaries to a batched tensor.
    """
    tensor_list = [crisp_obs_to_tensor(obs[0], copy=copy) for obs in observations]
    return torch.cat(tensor_list, dim=0)


def get_input_size_from_dict_space(obs_space) -> int:
    """
    Get the input size from the observation space.

    :param obs_space: The observation space
    :return: The input size
    """
    size = 0
    for space in obs_space.values():
        size += np.prod(space.shape)
    return size
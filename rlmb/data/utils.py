import torch
import numpy as np
import time

from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights
from collections import defaultdict

from rlmb.agents.rlpd.config import RLPD_Config


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
        if name in ["observation.state.joint", "task", "observation.state.target"]:
            continue
        elif 'image' in name:
            size += 128
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


def load_buffer_from_lerobot_dataset(dataset, buffer, num_episodes: int):
    """
    Load a dataset from the LeRobotDataset into a replay buffer.

    :param dataset: The dataset to load from
    :param buffer: The replay buffer to load into
    :param num_episodes: The number of episodes to load (default: None = load all)
    """
    BATCH_SIZE = 1024
    config = RLPD_Config()
    total_recorded_steps = len(dataset)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=8)
    obs = {}
    next_obs = {}
    new_episode = True

    for batch in dataloader:
        for i in range(1, batch['action'].shape[0]):
            # extract observations
            obs = next_obs
            next_obs['observation.state.cartesian'] = batch['observation.state.cartesian'][i].numpy()
            next_obs['observation.state.gripper'] = np.expand_dims(batch['observation.state.gripper'][i].numpy(), axis=0)
            next_obs['observation.images.primary'] = batch['observation.images.primary'][i].numpy() * 255.0
            next_obs['observation.images.wrist'] = batch['observation.images.wrist'][i].numpy() * 255.0

            action = batch['action'][i].numpy()

            # transpose image channel dim to last dim and change dtype
            next_obs['observation.images.primary'] = np.transpose(next_obs['observation.images.primary'], (1,2,0)).astype(np.uint8)
            next_obs['observation.images.wrist'] = np.transpose(next_obs['observation.images.wrist'], (1,2,0)).astype(np.uint8)

            #reward = - np.linalg.norm(action)
            reward = np.zeros(1)
            done = np.zeros(1, dtype=bool)

            # check for new episode
            if batch['frame_index'][i].item() == 0:
                reward = config.success_reward
                done = np.ones(1, dtype=bool)
                new_episode = True
                buffer.add(obs, obs, action, reward, done, {})
            
            if not new_episode:
                buffer.add(obs, next_obs, action, reward, done, {})
            else:
                new_episode = False
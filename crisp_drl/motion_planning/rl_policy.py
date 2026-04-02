"""RLPolicyPlanner — loads a trained SAC actor from a checkpoint directory
and wraps it as a BasePlanner for use in the hybrid planner test harness.

The planner formats the raw obs dict manually (matching ObservationFormatterWrapper
scaling) so that no DINOv2 encoder or CUDA wrapper stack is needed at inference
time (works on CPU with n_cameras=0 policies).
"""

import numpy as np
import torch
from pathlib import Path

from crisp_drl.agents.shared.networks_cleanrl import Actor, SharedEncoder
from crisp_drl.agents.shared.algorithm_config import Config
from crisp_drl.motion_planning.planners import BasePlanner


class RLPolicyPlanner(BasePlanner):
    """Wraps a trained SAC Actor + SharedEncoder checkpoint as a BasePlanner.

    Parameters
    ----------
    checkpoint_dir : str
        Path to checkpoint directory containing ``actor_state_dict.pth`` and
        ``shared_encoder_state_dict.pth``.
    config : Config, optional
        Algorithm config used when training the policy.  Must match the
        checkpoint architecture.  Defaults to Config().
    device : str
        Torch device, e.g. ``"cpu"`` or ``"cuda"``.  CPU is sufficient for
        the minimal test harness (no-vision policies).
    use_ft : bool
        Set True if the policy was trained with F/T sensor observations
        (adds 6 dims from ``observation.state.sensors_bota_ft_sensor``).
    """

    # Obs keys + slices + scaling — must match ObservationFormatterWrapper
    _OBS_KEYS_SCALES = [
        ("observation.previous.action",          (0, 2), 1000.0),
        ("observation.previous.error.cartesian", (0, 2), 1000.0),
        ("observation.velocity.cartesian",       (0, 2), 1000.0),
        ("observation.error.cartesian",          (0, 2), 1000.0),
    ]
    _FT_KEY_SCALE = ("observation.state.sensors_bota_ft_sensor", (0, 6), 0.1)

    def __init__(
        self,
        checkpoint_dir: str,
        config: Config = None,
        device: str = "cpu",
        use_ft: bool = False,
    ):
        if config is None:
            config = Config()
        self._device = torch.device(device)
        self._use_ft = use_ft

        ckpt = Path(checkpoint_dir)
        self._encoder = SharedEncoder(config).to(self._device)
        self._encoder.load_state_dict(
            torch.load(ckpt / "shared_encoder_state_dict.pth", map_location=self._device)
        )
        self._encoder.eval()

        self._actor = Actor(config).to(self._device)
        self._actor.load_state_dict(
            torch.load(ckpt / "actor_state_dict.pth", map_location=self._device)
        )
        self._actor.eval()

    def reset(self, goal_position: np.ndarray) -> None:
        pass  # stateless at episode level

    def plan(self, obs: dict) -> np.ndarray:
        keys_scales = list(self._OBS_KEYS_SCALES)
        if self._use_ft:
            keys_scales.append(self._FT_KEY_SCALE)

        parts = []
        for key, (lo, hi), scale in keys_scales:
            parts.append(obs[key][lo:hi].astype(np.float32) * scale)
        formatted = np.concatenate(parts)

        with torch.no_grad():
            x = torch.tensor(formatted, dtype=torch.float32, device=self._device).unsqueeze(0)
            encoded = self._encoder(x)
            action, _ = self._actor(encoded)
        return action.squeeze(0).cpu().numpy()

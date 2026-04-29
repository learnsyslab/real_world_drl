"""2x4 variant of LEGO 6DoF PE wrapper.

Keeps the same reset state machine as InsertionWrapperLegoPE and only switches
the PE backend to coarse-only (no tracker) for 2x4 bring-up.
"""

from crisp_drl.agents.shared.algorithm_config import Config
from crisp_drl.agents.shared.insertion_env_config import LegoConfig2x4
from crisp_drl.agents.shared.insertion_wrapper_lego import InsertionWrapperLegoPE


class InsertionWrapperLego2x4PE(InsertionWrapperLegoPE):
    def __init__(
        self,
        env,
        alg_config: Config,
        env_config: LegoConfig2x4,
        **kwargs,
    ):
        # Keep parent state machine unchanged; only disable tracker for 2x4.
        super().__init__(
            env,
            alg_config=alg_config,
            env_config=env_config,
            use_tracker=False,
            **kwargs,
        )

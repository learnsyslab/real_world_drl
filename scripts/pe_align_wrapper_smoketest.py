"""Smoke test: instantiate InsertionWrapper3DoFRotZ with pe_align_gripper enabled.

This script avoids initializing pose estimation internals by setting
`use_pose_estimation=False`. It's a minimal smoke test that ensures the
constructor and attribute are available.
"""
from crisp_drl.agents.shared.insertion_wrapper import InsertionWrapper3DoFRotZ
from crisp_drl.agents.shared.algorithm_config import Config


class DummyEnv:
    pass


def main():
    cfg = Config()
    env = DummyEnv()
    wrapper = InsertionWrapper3DoFRotZ(env, cfg, use_pose_estimation=False, pe_align_gripper=True)
    print("pe_align_gripper:", wrapper.pe_align_gripper)


if __name__ == "__main__":
    main()

import os
import numpy as np
from pathlib import Path
from gymnasium import spaces
import torch
import pickle
from crisp_drl.agents.shared.algorithm_config import Config
from crisp_drl.data.buffers_cleanrl import ReplayBufferGpuWithPerfectActions

# Combine per-subfolder data and cross-folder into single rollout buffers with additional field perfect_action

config = Config()
exp_folders_to_skip = {
    "20251204-095036_0.95_45",
    "20251204-095627_0.33_99",
    "20251204-100213_0.999_26",
    "20251204-101910_0.9_46",
    "20251204-102434_0.8_92",
    "20251205-083116_0.33_100",
    "20251205-084554_0.8_92",
    "20251205-085351_0.9_56",
    "20251205-090548_0.95_39",
    "20251205-091208_0.999_24",
    "20251212-092812_0.999_18",
    "20251212-093216_0.33_100",
    "20251212-095949_0.33_100",
    "20251212-100257_0.999_22",
    "20251212-115108_0.999_23",
    "20251212-115351_0.33_100",
    "20251212-130027_0.33_100",  # sparse + delta place + action magnitude
    "20251212-130201_0.999_24",  # sparse + delta place + action magnitude
    "20251215-105714_0.999_30",  # sparse only
    "20251215-110132_0.33_100",  # sparse only
    "20251215-112134_0.33_100",  # sparse + delta place
    "20251215-112238_0.999_17",  # sparse + delta place
    "20251215-120456_0.999_15",  # sparse + delta place + action magnitude
    "20251215-120723_0.33_100",  # sparse + delta place + action magnitude
    "20251215-134429_0.33_99",  # sparse + delta simple + action magnitude
    "20251215-134537_0.999_24",  # sparse + delta simple + action magnitude
    "20251215-144507_0.33_100",  # sparse only
    "20251215-144631_0.999_28",  # sparse only
    "20251215-151319_0.5_100",  # sparse only
    "20251215-152854_0.999_23",  # sparse only
    "20251215-161436_0.999_18",
    "20251215-161550_0.9_48",
    "20251215-161657_0.8_94",
    "20251215-161813_0.33_100",
    # start of 25-batch runs
    "20251222-121602_0.33_100",
    "20251222-121615_0.8_92",
    "20251222-121639_0.9_60",
    "20251222-121709_0.999_12",
    "20251222-133328_0.33_100",
    "20251222-133341_0.8_88",
    "20251222-133401_0.9_68",
    "20251222-133425_0.999_16",
    "20251222-145841_0.33_100",
    "20251222-145854_0.33_100",
    "20251222-145907_0.999_24",
    "20251222-145938_0.999_16",
    # batch 100 runs
    "20251223-120851_0.33_100",
    "20251223-120945_0.8_79",
    "20251223-121121_0.9_47",
    "20251223-121317_0.95_45",
    "20251223-121502_0.999_24",
    # batch 500 runs
    "20251223-132427_0.0_100",
    "20251223-135234_0.33_99",
    "20251223-135857_0.8_84",  # l=42.37
    "20251223-140641_0.9_55",  # l=49.34
    "20251223-141653_0.95_37",  # l=49.59
    "20251223-142728_0.999_21",  # l=45.76
    # batch 300 runs
    "20251226-001530_0.75_94",  # l=38.76
    "20251226-001826_0.85_73",  # l=48.12
    # batch 500, extened safety box
    "20251226-181302_0.85_73",  # l=48.65
    # batch 400, length 150
    "20251230-182128_0.85_87",  # l=63.82
    # batch 400, length 200
    "20251231-103020_0.9_77",  # l=77.66
    "20251231-135935_0.88_82",  # l= 71.63
    # Better env
    # batch 300, length 150, 2 obs
    "20260126-113312_0.8_67",  # l=99.30
    # 8 obs
    "20260128-093443_0.85_50",  # l=107.41
    "20260128-093554_0.85_54",  # l=109.64
    "20260128-094610_0.8_71",  # l=103.05
    "20260128-094612_0.8_72",  # l=101.72
    "20260128-100707_0.75_87",  # l=87.97
    "20260128-100711_0.75_84",  # l=88.79
    # Env v5: larger grasp randomisation following ellipse in training
    "20260202-102044_0.75_86",  # l=89.33
    "20260202-102126_0.75_89",  # l=85.86
    # Env v6: slightly smaller grasp randomisation and box shape in training
    "20260202-123146_0.75_88",  # l=85.56
    "20260202-123139_0.75_86",  # l=86.76
    # Env v7: grasp randomisation +-0.75mm z, 1.5mm x, box shape in training
    "20260203-133410_0.75_86",  # l=86.23
    "20260203-133426_0.75_86",  # l=90.48
    # Env v8: with force torque sensor observations
    "20260204-101944_0.75_88",  # l=89.66
    "20260204-101959_0.75_85",  # l=88.43
    # Big ablation env v9
    "20260213-174430_0.0_95",
    "20260213-174527_0.0_96",
    "20260213-174552_0.0_95",
    "20260213-174858_0.3_93",
    "20260213-175023_0.3_95",
    "20260213-175043_0.3_93",
    "20260213-175332_0.6_93",
    "20260213-175424_0.6_93",
    "20260213-175448_0.6_93",
    "20260213-175821_0.7_89",
    "20260213-175905_0.7_91",
    "20260213-175930_0.7_91",
    "20260213-180359_0.75_84",
    "20260213-180451_0.75_86",
    "20260213-180507_0.75_84",
    "20260213-181026_0.8_77",
    "20260213-181118_0.8_72",
    "20260213-181135_0.8_75",
    "20260213-181732_0.85_48",
    "20260213-181839_0.85_56",
    "20260213-181840_0.85_53",
    "20260213-182527_0.9_20",
    "20260213-182626_0.9_22",
    "20260213-182630_0.9_26",
    "20260213-183417_0.95_05",
    "20260213-183512_0.95_07",
    "20260213-183521_0.95_07",
    "20260213-184243_1.0_01",
    "20260213-184335_1.0_02",
    "20260213-184358_1.0_01",
    ## Ablation on sensors
    "20260218-115652_0.8_72",
    "20260218-115718_0.8_76",
    "20260218-115752_0.8_75",
    ## Ablation on pe accuracy - 0.0015m
    "20260224-184848_0.8_76",
    "20260224-184904_0.8_74",
    "20260224-184919_0.8_78",
    # 0.002m - 180 steps
    "20260225-134739_0.75_77",
    "20260225-134745_0.75_77",
    "20260225-134753_0.75_78",
    "20260225-145243_0.8_74",
    "20260225-145248_0.8_75",
    "20260225-145250_0.8_71",
    # 0.0025m - 240 steps
    "20260225-155431_0.6_75",
    "20260225-155605_0.6_74",
    "20260225-155610_0.6_73",
    "20260225-163910_0.8_68",
    "20260225-163958_0.8_68",
    "20260225-164010_0.8_66",
    # 0.003m - 330 steps
    "20260225-195546_0.8_51",
    "20260225-195708_0.8_52",
    "20260225-195718_0.8_50",
    "20260225-203457_0.4_61",
    "20260225-203543_0.4_58",
    "20260225-203549_0.4_58",
    ## Ablation on pe accuracy, now with z-randomisation 0.5 - 0.0015m
    "20260227-184207_0.8_76",
    "20260228-003003_0.8_76",
    "20260228-010759_0.8_76",
    # 0.002m - 180 steps
    "20260227-184324_0.75_79",
    "20260228-003032_0.75_76",
    "20260228-010817_0.75_78",
    "20260227-184407_0.8_72",
    "20260228-003049_0.8_74",
    "20260228-011544_0.8_74",
    # 0.0025m - 240 steps
    "20260227-184508_0.6_70",
    "20260228-003102_0.6_73",
    "20260228-010359_0.6_74",
    "20260227-184543_0.8_67",
    "20260228-003120_0.8_64",
    "20260228-012301_0.8_65",
    # 0.003m - 330 steps
    "20260227-184651_0.4_57",
    "20260228-003138_0.4_60",
    "20260228-010043_0.4_62",
    "20260227-184736_0.8_51",
    "20260228-003156_0.8_52",
    "20260228-013252_0.8_53",
    ## Ablation on pe accuracy, now with z-randomisation 0.33 - 0.0015m
    "20260301-121541_0.8_76",
    "20260301-125427_0.8_78",
    "20260301-132938_0.8_77",
    # 0.002m - 180 steps
    "20260301-121550_0.75_75",
    "20260301-125436_0.75_79",
    "20260301-133014_0.75_77",
    "20260301-125653_0.8_72",
    "20260301-133952_0.8_74",
    "20260301-141136_0.8_72",
    # 0.0025m - 240 steps
    "20260301-131848_0.6_71",
    "20260301-124903_0.6_74",
    "20260301-121609_0.6_74",
    "20260301-121620_0.8_63",
    "20260301-130833_0.8_64",
    "20260301-135456_0.8_65",
    # 0.003m - 330 steps
    "20260301-124523_0.4_60",
    "20260301-131113_0.4_57",
    "20260301-121629_0.4_56",
    "20260301-121644_0.8_51",
    "20260301-131635_0.8_48",
    "20260301-140541_0.8_50",
}

exp_name = "v9_1cft5d_halfrot"
EXPECTED_MAX_LEN = 150
N_SKIP = 0
N_SEEDS = 3
FIRST_SEED = 1
parent_folder = "rollout_data/collect_data"
rollouts = [
    "20260406-094652_0.8_81",
    "20260406-102138_0.8_80",
    "20260406-105605_0.8_81",
]

for SEED in range(FIRST_SEED, FIRST_SEED + N_SEEDS):
    exp_path = Path(parent_folder) / rollouts[SEED - FIRST_SEED]
    for SUBSET_SIZE in [2000]:
        global_buffer_file_name = (
            f"replay_buffer_{SUBSET_SIZE}{exp_name}_s{SEED}.joblib"
        )

        obs_features = None  # [
        #     (0, 8),  # 8 nonvision non ft features
        #     (8, 14),  # force torque
        #     (14, 398),  # cam 1
        #     # (398, 782),  # cam 2
        # ]

        OBS_DIM = (
            sum(x[1] - x[0] for x in obs_features)  # pyright: ignore[reportGeneralTypeIssues]
            if obs_features is not None
            else config.actor_nonvision_input_dim
            + config.n_cameras * config.vision_head_input_dim
        )

        # N_COMPLETION_TARGET = 180

        # 0) Load all data from subfolders
        all_run_data = {}
        print(f"Processing folder: {exp_path.name}")

        all_run_data[exp_path.name] = []
        i = 0
        # n_completed = 0
        for run_path in sorted(os.listdir(exp_path)):
            run_file = exp_path / run_path
            if (
                not run_file.is_file()
                or run_path.endswith("_info.pkl")
                or run_path.endswith(".joblib")
            ):
                continue

            run_data = np.load(run_file, allow_pickle=True)
            # if n_completed >= N_COMPLETION_TARGET and run_data["terminated"]:
            #     continue
            # if (
            #     len(all_run_data[exp_folder]) - n_completed
            #     >= (SUBSET_SIZE - N_COMPLETION_TARGET)
            #     and not run_data["terminated"]
            # ):
            #     continue
            # if run_data["terminated"]:
            #     n_completed += 1

            i += 1
            if i <= N_SKIP:
                continue
            with open(run_file.with_name(run_file.stem + "_info.pkl"), "rb") as f:
                info_data = pickle.load(f)
            for info in info_data:
                custom_events = info.get("custom_events", [])
                event_names = [event[1] for event in custom_events]
                if (
                    "E_ROLLOUT_UNUSABLE" in event_names
                    or "E_CONTROLLER_ISSUE" in event_names
                    or "E_TORQUE" in event_names
                ):
                    print(f"  Skipping unusable rollout: {run_path}")
                    continue

            all_run_data[exp_path.name].append(run_data)
            if len(all_run_data[exp_path.name]) >= SUBSET_SIZE:
                break

        # separate image encoders from buffer and move into actor?
        #  => decide in actor whether to train image encoder or not
        # online VAE training? -> probably does not do much, no fundamentally new data
        #  => maybe continue training with policy gradient, similar to no pre-training
        # Pipeline:
        #    [1) VAE vs no VAE]
        #     2) Actor+Critic [image encoder continue-training vs fixed]
        #     3) RL [image encoder continue-training vs fixed]

        # 1) compute buffer sizes
        buffer_sizes = {}
        max_len = 0
        for exp_folder, run_datas in all_run_data.items():
            total_size = 0
            for run_data in run_datas:  # [: len(run_datas) // 2]:
                total_size += len(run_data["observations"])
                max_len = max(max_len, len(run_data["observations"]))
            buffer_sizes[exp_folder] = total_size
        global_buffer_size = sum(buffer_sizes.values())
        if EXPECTED_MAX_LEN is not None:
            assert EXPECTED_MAX_LEN == max_len - 1, (
                f"Expected max rollout length {EXPECTED_MAX_LEN}, got {max_len - 1}"
            )

        # 2) create buffers
        buffers = {}
        for exp_folder, size in buffer_sizes.items():
            buffers[exp_folder] = ReplayBufferGpuWithPerfectActions(
                action_space=spaces.Box(
                    low=-np.inf, high=np.inf, shape=(config.actor_output_dim,)
                ),
                n_step_return=1,
                gamma=config.gamma,
                device="cuda",
                buffer_size=size,
                observation_dim=OBS_DIM,
            )
        global_buffer = ReplayBufferGpuWithPerfectActions(
            action_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(config.actor_output_dim,)
            ),
            n_step_return=1,
            gamma=config.gamma,
            device="cuda",
            buffer_size=global_buffer_size,
            observation_dim=OBS_DIM,
        )

        n_rollouts_total = 0
        # 3) fill buffers
        for exp_folder, run_datas in all_run_data.items():
            # print(f"Filling buffers for folder: {exp_folder}")
            for i, run_data in enumerate(run_datas):  # [: len(run_datas) // 2]:
                # print(f"  Processing rollout {i + 1} / {len(run_datas)}")
                all_observations = run_data["observations"]
                all_actions = run_data["actions"]
                all_rewards = run_data["rewards"]
                all_perfect_actions = run_data["perfect_actions"]
                terminated = bool(run_data["terminated"])

                # Select only the sub-ranges specified in obs_features
                def select_obs_features(obs_arr):
                    # obs_arr: shape (..., obs_dim)
                    # returns: shape (..., sum of selected dims)
                    return np.concatenate(
                        [obs_arr[..., start:end] for (start, end) in obs_features],  # pyright: ignore[reportGeneralTypeIssues]
                        axis=-1,
                    )

                selected_observations = (
                    select_obs_features(np.array(all_observations))
                    if obs_features is not None
                    else np.array(all_observations)
                )
                obs = list(map(torch.from_numpy, selected_observations))
                act = list(map(torch.from_numpy, all_actions))
                perf_act = list(map(torch.from_numpy, all_perfect_actions))

                global_buffer.add_rollout(
                    obs=obs,
                    action=act,
                    reward=all_rewards,
                    perfect_action=perf_act,
                    terminated=terminated,
                )
                buffers[exp_folder].add_rollout(
                    obs=obs,
                    action=act,
                    reward=all_rewards,
                    perfect_action=perf_act,
                    terminated=terminated,
                )
                n_rollouts_total += 1
        print(f"Total number of rollouts added: {n_rollouts_total}")
        # 4) save buffers
        # for exp_folder, buffer in buffers.items():
        #     save_path = Path(base_folder) / exp_folder
        #     buffer.save_buffer(save_path)
        global_save_path = Path(parent_folder) / global_buffer_file_name
        global_buffer.save_buffer(global_save_path)

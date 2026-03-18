#!/bin/bash


# Check if argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <seed>"
    echo "Example: $0 1"
    exit 1
fi

SEED=$1

# pixi run -e jazzy python scripts/run_sac_pretraining.py --run_name 1cft_d200v9_15_s${SEED} --pre_train rollout_data/collect_data/replay_buffer_200v9_1cft_15_s${SEED}.joblib --seed ${SEED} --utd_ratio 15
# pixi run -e jazzy python scripts/run_sac_pretraining.py --run_name 1cft_d300v9_15_s${SEED} --pre_train rollout_data/collect_data/replay_buffer_300v9_1cft_15_s${SEED}.joblib --seed ${SEED} --utd_ratio 10
# pixi run -e jazzy python scripts/run_sac_pretraining.py --run_name 1cft_d400v9_15_s${SEED} --pre_train rollout_data/collect_data/replay_buffer_400v9_1cft_15_s${SEED}.joblib --seed ${SEED} --utd_ratio 10

# pixi run -e jazzy python scripts/run_sac_pretraining.py --run_name 1cft_d300v9_20_s${SEED} --pre_train rollout_data/collect_data/replay_buffer_300v9_1cft_20_s${SEED}.joblib --seed ${SEED} --utd_ratio 10
# pixi run -e jazzy python scripts/run_sac_pretraining.py --run_name 1cft_d400v9_20_s${SEED} --pre_train rollout_data/collect_data/replay_buffer_400v9_1cft_20_s${SEED}.joblib --seed ${SEED} --utd_ratio 10
# pixi run -e jazzy python scripts/run_sac_pretraining.py --run_name 1cft_d500v9_20_s${SEED} --pre_train rollout_data/collect_data/replay_buffer_500v9_1cft_20_s${SEED}.joblib --seed ${SEED} --utd_ratio 10

# pixi run -e jazzy python scripts/run_sac_pretraining.py --run_name 1cft_d300v9_20_75_s${SEED} --pre_train rollout_data/collect_data/replay_buffer_300v9_1cft_20_75_s${SEED}.joblib --seed ${SEED} --utd_ratio 10
# pixi run -e jazzy python scripts/run_sac_pretraining.py --run_name 1cft_d400v9_20_75_s${SEED} --pre_train rollout_data/collect_data/replay_buffer_400v9_1cft_20_75_s${SEED}.joblib --seed ${SEED} --utd_ratio 10
# pixi run -e jazzy python scripts/run_sac_pretraining.py --run_name 1cft_d500v9_20_75_s${SEED} --pre_train rollout_data/collect_data/replay_buffer_500v9_1cft_20_75_s${SEED}.joblib --seed ${SEED} --utd_ratio 10



# pixi run -e jazzy python scripts/run_sac_pretraining.py --run_name 1cft_d400v9_25_s${SEED} --pre_train rollout_data/collect_data/replay_buffer_400v9_1cft_25_s${SEED}.joblib --seed ${SEED} --utd_ratio 0.9
# pixi run -e jazzy python scripts/run_sac_pretraining.py --run_name 1cft_d500v9_25_s${SEED} --pre_train rollout_data/collect_data/replay_buffer_500v9_1cft_25_s${SEED}.joblib --seed ${SEED} --utd_ratio 0.9
# pixi run -e jazzy python scripts/run_sac_pretraining.py --run_name 1cft_d600v9_25_s${SEED} --pre_train rollout_data/collect_data/replay_buffer_600v9_1cft_25_s${SEED}.joblib --seed ${SEED} --utd_ratio 0.9

# pixi run -e jazzy python scripts/run_sac_pretraining.py --run_name 1cft_d400v9_25_60_s${SEED} --pre_train rollout_data/collect_data/replay_buffer_400v9_1cft_25_60_s${SEED}.joblib --seed ${SEED} --utd_ratio 8
# pixi run -e jazzy python scripts/run_sac_pretraining.py --run_name 1cft_d500v9_25_60_s${SEED} --pre_train rollout_data/collect_data/replay_buffer_500v9_1cft_25_60_s${SEED}.joblib --seed ${SEED} --utd_ratio 8
# pixi run -e jazzy python scripts/run_sac_pretraining.py --run_name 1cft_d600v9_25_60_s${SEED} --pre_train rollout_data/collect_data/replay_buffer_600v9_1cft_25_60_s${SEED}.joblib --seed ${SEED} --utd_ratio 8


pixi run -e jazzy python scripts/run_sac_pretraining.py --run_name 1cft_d500v9_30_s${SEED} --pre_train rollout_data/collect_data/replay_buffer_500v9_1cft_30_s${SEED}.joblib --seed ${SEED} --utd_ratio 0.965
pixi run -e jazzy python scripts/run_sac_pretraining.py --run_name 1cft_d600v9_30_s${SEED} --pre_train rollout_data/collect_data/replay_buffer_600v9_1cft_30_s${SEED}.joblib --seed ${SEED} --utd_ratio 0.965
pixi run -e jazzy python scripts/run_sac_pretraining.py --run_name 1cft_d700v9_30_s${SEED} --pre_train rollout_data/collect_data/replay_buffer_700v9_1cft_30_s${SEED}.joblib --seed ${SEED} --utd_ratio 0.965
pixi run -e jazzy python scripts/run_sac_pretraining.py --run_name 1cft_d800v9_30_s${SEED} --pre_train rollout_data/collect_data/replay_buffer_800v9_1cft_30_s${SEED}.joblib --seed ${SEED} --utd_ratio 0.965


# pixi run -e jazzy python scripts/run_sac_pretraining.py --run_name 1cft_d500v9_30_40_s${SEED} --pre_train rollout_data/collect_data/replay_buffer_500v9_1cft_30_40_s${SEED}.joblib --seed ${SEED} --utd_ratio 10
# pixi run -e jazzy python scripts/run_sac_pretraining.py --run_name 1cft_d600v9_30_40_s${SEED} --pre_train rollout_data/collect_data/replay_buffer_600v9_1cft_30_40_s${SEED}.joblib --seed ${SEED} --utd_ratio 10
# pixi run -e jazzy python scripts/run_sac_pretraining.py --run_name 1cft_d700v9_30_40_s${SEED} --pre_train rollout_data/collect_data/replay_buffer_700v9_1cft_30_40_s${SEED}.joblib --seed ${SEED} --utd_ratio 10
# pixi run -e jazzy python scripts/run_sac_pretraining.py --run_name 1cft_d800v9_30_40_s${SEED} --pre_train rollout_data/collect_data/replay_buffer_800v9_1cft_30_40_s${SEED}.joblib --seed ${SEED} --utd_ratio 10

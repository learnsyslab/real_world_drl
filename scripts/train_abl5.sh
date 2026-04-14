#!/bin/bash


# Check if argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <seed>"
    echo "Example: $0 1"
    exit 1
fi

SEED=$1



# pixi run -e jazzy python scripts/run_sac_pretraining.py --run_name 1cft5d_d500v9_s${SEED} --pre_train rollout_data/collect_data/replay_buffer_500v9_1cft5d_s${SEED}.joblib --seed ${SEED} --utd_ratio 8
# pixi run -e jazzy python scripts/run_sac_pretraining.py --run_name 1cft5d_d600v9_s${SEED} --pre_train rollout_data/collect_data/replay_buffer_600v9_1cft5d_s${SEED}.joblib --seed ${SEED} --utd_ratio 8
# pixi run -e jazzy python scripts/run_sac_pretraining.py --run_name 1cft5d_d700v9_s${SEED} --pre_train rollout_data/collect_data/replay_buffer_700v9_1cft5d_s${SEED}.joblib --seed ${SEED} --utd_ratio 8
# pixi run -e jazzy python scripts/run_sac_pretraining.py --run_name 1cft5d_d800v9_s${SEED} --pre_train rollout_data/collect_data/replay_buffer_800v9_1cft5d_s${SEED}.joblib --seed ${SEED} --utd_ratio 8

pixi run -e jazzy python scripts/run_sac_pretraining.py --run_name 1cft5d_halfrot_d2000v9_s${SEED} --pre_train rollout_data/collect_data/replay_buffer_2000v9_1cft5d_halfrot_s${SEED}.joblib --seed ${SEED} --utd_ratio 5


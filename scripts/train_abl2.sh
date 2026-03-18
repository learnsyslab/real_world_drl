#!/bin/bash


# Check if argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <seed>"
    echo "Example: $0 1"
    exit 1
fi

SEED=$1

pixi run -e jazzy python scripts/run_sac_pretraining.py --run_name 1c_d100v9_s${SEED} --pre_train rollout_data/collect_data/replay_buffer_100v9_1c_s${SEED}.joblib --seed ${SEED} --utd_ratio 20 --n_cameras 1 --actor_nonvision_input_dim 8
pixi run -e jazzy python scripts/run_sac_pretraining.py --run_name 1cft_d100v9_s${SEED} --pre_train rollout_data/collect_data/replay_buffer_100v9_1cft_s${SEED}.joblib --seed ${SEED} --utd_ratio 20 --n_cameras 1 --actor_nonvision_input_dim 14
pixi run -e jazzy python scripts/run_sac_pretraining.py --run_name 1c_d150v9_s${SEED} --pre_train rollout_data/collect_data/replay_buffer_150v9_1c_s${SEED}.joblib --seed ${SEED} --utd_ratio 20 --n_cameras 1 --actor_nonvision_input_dim 8
pixi run -e jazzy python scripts/run_sac_pretraining.py --run_name 1cft_d150v9_s${SEED} --pre_train rollout_data/collect_data/replay_buffer_150v9_1cft_s${SEED}.joblib --seed ${SEED} --utd_ratio 20 --n_cameras 1 --actor_nonvision_input_dim 14



# pixi run -e jazzy python scripts/run_sac_pretraining.py --run_name 1c_d200v9_s${SEED} --pre_train rollout_data/collect_data/replay_buffer_200v9_1c_s${SEED}.joblib --seed ${SEED} --utd_ratio 30 --n_cameras 1 --actor_nonvision_input_dim 8
# pixi run -e jazzy python scripts/run_sac_pretraining.py --run_name 1c_d300v9_s${SEED} --pre_train rollout_data/collect_data/replay_buffer_300v9_1c_s${SEED}.joblib --seed ${SEED} --utd_ratio 20 --n_cameras 1 --actor_nonvision_input_dim 8
# pixi run -e jazzy python scripts/run_sac_pretraining.py --run_name 1c_d400v9_s${SEED} --pre_train rollout_data/collect_data/replay_buffer_400v9_1c_s${SEED}.joblib --seed ${SEED} --utd_ratio 20 --n_cameras 1 --actor_nonvision_input_dim 8
# pixi run -e jazzy python scripts/run_sac_pretraining.py --run_name 2c_d200v9_s${SEED} --pre_train rollout_data/collect_data/replay_buffer_200v9_2c_s${SEED}.joblib --seed ${SEED} --utd_ratio 30 --n_cameras 2 --actor_nonvision_input_dim 8
# pixi run -e jazzy python scripts/run_sac_pretraining.py --run_name 2c_d300v9_s${SEED} --pre_train rollout_data/collect_data/replay_buffer_300v9_2c_s${SEED}.joblib --seed ${SEED} --utd_ratio 20 --n_cameras 2 --actor_nonvision_input_dim 8
# pixi run -e jazzy python scripts/run_sac_pretraining.py --run_name 2c_d400v9_s${SEED} --pre_train rollout_data/collect_data/replay_buffer_400v9_2c_s${SEED}.joblib --seed ${SEED} --utd_ratio 20 --n_cameras 2 --actor_nonvision_input_dim 8
# pixi run -e jazzy python scripts/run_sac_pretraining.py --run_name 1cft_d200v9_s${SEED} --pre_train rollout_data/collect_data/replay_buffer_200v9_1cft_s${SEED}.joblib --seed ${SEED} --utd_ratio 30 --n_cameras 1 --actor_nonvision_input_dim 14
# pixi run -e jazzy python scripts/run_sac_pretraining.py --run_name 1cft_d300v9_s${SEED} --pre_train rollout_data/collect_data/replay_buffer_300v9_1cft_s${SEED}.joblib --seed ${SEED} --utd_ratio 20 --n_cameras 1 --actor_nonvision_input_dim 14
# pixi run -e jazzy python scripts/run_sac_pretraining.py --run_name 1cft_d400v9_s${SEED} --pre_train rollout_data/collect_data/replay_buffer_400v9_1cft_s${SEED}.joblib --seed ${SEED} --utd_ratio 20 --n_cameras 1 --actor_nonvision_input_dim 14


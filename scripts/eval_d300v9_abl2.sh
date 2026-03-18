#!/bin/bash

# Check if argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <seed>"
    echo "Example: $0 1"
    exit 1
fi

SEED=$1


# Run eval_actor.py for different pretrain stages and camera configurations
# PRETRAIN_STAGES=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 20 25 30)
# for stage in "${PRETRAIN_STAGES[@]}"; do
#     echo "Stage (1) ${stage}"

#     pixi run -e jazzy python crisp_drl/scripts/eval_actor.py --run_name 1c_d200v9_s${SEED}/pretrain_${stage}   --n_cameras 1 
#     pixi run -e jazzy python crisp_drl/scripts/eval_actor.py --run_name 2c_d200v9_s${SEED}/pretrain_${stage}    --n_cameras 2 
#     pixi run -e jazzy python crisp_drl/scripts/eval_actor.py --run_name 1cft_d200v9_s${SEED}/pretrain_${stage}    --n_cameras 1 --use_ft

# done

# Run eval_actor.py for different pretrain stages and camera configurations
PRETRAIN_STAGES=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 20)
for stage in "${PRETRAIN_STAGES[@]}"; do

    echo "Stage (2) ${stage}"

    # pixi run -e jazzy python crisp_drl/scripts/eval_actor.py --run_name 1c_d300v9_s${SEED}/pretrain_${stage}    --n_cameras 1 
    # pixi run -e jazzy python crisp_drl/scripts/eval_actor.py --run_name 1c_d400v9_s${SEED}/pretrain_${stage}    --n_cameras 1 
    # pixi run -e jazzy python crisp_drl/scripts/eval_actor.py --run_name 2c_d300v9_s${SEED}/pretrain_${stage}    --n_cameras 2 
    # pixi run -e jazzy python crisp_drl/scripts/eval_actor.py --run_name 2c_d400v9_s${SEED}/pretrain_${stage}    --n_cameras 2 
    # pixi run -e jazzy python crisp_drl/scripts/eval_actor.py --run_name 1cft_d300v9_s${SEED}/pretrain_${stage}    --n_cameras 1 --use_ft
    # pixi run -e jazzy python crisp_drl/scripts/eval_actor.py --run_name 1cft_d400v9_s${SEED}/pretrain_${stage}    --n_cameras 1 --use_ft

    pixi run -e jazzy python crisp_drl/scripts/eval_actor.py --run_name 1c_d100v9_s${SEED}/pretrain_${stage}    --n_cameras 1 
    pixi run -e jazzy python crisp_drl/scripts/eval_actor.py --run_name 1c_d150v9_s${SEED}/pretrain_${stage}    --n_cameras 1 
    pixi run -e jazzy python crisp_drl/scripts/eval_actor.py --run_name 1cft_d100v9_s${SEED}/pretrain_${stage}    --n_cameras 1 --use_ft
    pixi run -e jazzy python crisp_drl/scripts/eval_actor.py --run_name 1cft_d150v9_s${SEED}/pretrain_${stage}    --n_cameras 1 --use_ft



done

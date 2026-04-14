#!/bin/bash

# Check if argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <seed>"
    echo "Example: $0 1"
    exit 1
fi

SEED=$1


# Run eval_actor_3dof.py for different pretrain stages and camera configurations
PRETRAIN_STAGES=(1 2 3 4 5 6 7 8 9 10)
for stage in "${PRETRAIN_STAGES[@]}"; do

    echo "Stage (1) ${stage}"

    pixi run -e jazzy python crisp_drl/scripts/eval_actor_3dof.py --run_name 1cft3d_d300v9_s${SEED}/pretrain_${stage} --use_ft
    pixi run -e jazzy python crisp_drl/scripts/eval_actor_3dof.py --run_name 1cft3d_d400v9_s${SEED}/pretrain_${stage} --use_ft
    pixi run -e jazzy python crisp_drl/scripts/eval_actor_3dof.py --run_name 1cft3d_d500v9_s${SEED}/pretrain_${stage} --use_ft



done


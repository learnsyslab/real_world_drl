#!/bin/bash

# Check if argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <seed>"
    echo "Example: $0 1"
    exit 1
fi

SEED=$1


# Run eval_actor_3dof.py for different pretrain stages and camera configurations
PRETRAIN_STAGES=(0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 1 2 3 4 5)
for stage in "${PRETRAIN_STAGES[@]}"; do

    echo "Stage (1) ${stage}"

    pixi run -e jazzy python crisp_drl/scripts/eval_actor_5dof.py --run_name 1cft5d_halfrot_d2000v9_s${SEED}/pretrain_${stage} --use_ft
    # pixi run -e jazzy python crisp_drl/scripts/eval_actor_5dof.py --run_name 1cft5d_d500v9_s${SEED}/pretrain_${stage} --use_ft
    # pixi run -e jazzy python crisp_drl/scripts/eval_actor_5dof.py --run_name 1cft5d_d600v9_s${SEED}/pretrain_${stage} --use_ft
    # pixi run -e jazzy python crisp_drl/scripts/eval_actor_5dof.py --run_name 1cft5d_d700v9_s${SEED}/pretrain_${stage} --use_ft
    # pixi run -e jazzy python crisp_drl/scripts/eval_actor_5dof.py --run_name 1cft5d_d800v9_s${SEED}/pretrain_${stage} --use_ft



done


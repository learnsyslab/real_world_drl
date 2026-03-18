#!/bin/bash

# Configuration
PRETRAIN_STAGES=(2 4 6 7 8 9 11 12 13 14)
SEEDS=(1 2)

# Check if argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <checkpoint_name>"
    echo "Example: $0 1cam_128_16_full_d200v8_75_s1"
    exit 1
fi

CHECKPOINT=$1

echo "Evaluating checkpoint: $CHECKPOINT"

# Run eval_actor.py for different pretrain stages and camera configurations
for stage in "${PRETRAIN_STAGES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "Running: ${CHECKPOINT}/pretrain_${stage} ${seed}"
        pixi run -e jazzy python crisp_drl/scripts/eval_actor.py ${CHECKPOINT}/pretrain_${stage} ${seed}
    done
done

echo "Evaluation complete for $CHECKPOINT"

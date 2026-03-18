#!/bin/bash

# Configuration
PROBABILITIES=(7 75 8 85)

# Check if argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <train_seed>"
    echo "Example: $0 1"
    exit 1
fi

TRAIN_SEED=$1

echo "Evaluating d300v9 checkpoints for training seed: $TRAIN_SEED"

# Run eval_checkpoints.sh for each probability
for prob in "${PROBABILITIES[@]}"; do
    RUN_NAME="1cam_ft_128_16_full_d300v9_${prob}_s${TRAIN_SEED}"
    echo "=========================================="
    echo "Processing run: $RUN_NAME"
    echo "=========================================="
    bash "$(dirname "$0")/eval_checkpoints.sh" "$RUN_NAME"
done

echo "All evaluations complete for training seed $TRAIN_SEED"

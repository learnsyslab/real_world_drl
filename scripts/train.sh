#!/bin/bash

# Configuration
PROBABILITIES=(0 3 6 7 75 8 85 9 95 100)

# Check if argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <seed>"
    echo "Example: $0 1"
    exit 1
fi

SEED=$1

echo "Evaluating checkpoint: $CHECKPOINT"

# Run eval_actor.py for different pretrain stages and camera configurations
for prob in "${PROBABILITIES[@]}"; do
    echo "Running ${SEED}"
    pixi run -e jazzy python scripts/run_sac_pretraining.py --run_name 1cam_ft_128_16_full_d300v9_${prob}_s${SEED} --pre_train rollout_data/collect_data/replay_buffer_300v9_${prob}_s${SEED}.joblib --seed ${SEED}
done


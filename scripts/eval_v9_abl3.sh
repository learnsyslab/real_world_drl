#!/bin/bash

Check if argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <seed>"
    echo "Example: $0 1"
    exit 1
fi

SEED=$1


# Run eval_actor.py for different pretrain stages and camera configurations
# PRETRAIN_STAGES=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
# for stage in "${PRETRAIN_STAGES[@]}"; do

#     echo "Stage (1) ${stage}"

#     pixi run -e jazzy python crisp_drl/scripts/eval_actor.py --run_name 1cft_d200v9_15_s${SEED}/pretrain_${stage} --use_ft



# done

# PRETRAIN_STAGES=(1 2 3 4 5 6 7 8 9 10)
# for stage in "${PRETRAIN_STAGES[@]}"; do

#     echo "Stage (2) ${stage}"

#     pixi run -e jazzy python crisp_drl/scripts/eval_actor.py --run_name 1cft_d300v9_15_s${SEED}/pretrain_${stage} --use_ft
#     pixi run -e jazzy python crisp_drl/scripts/eval_actor.py --run_name 1cft_d300v9_20_s${SEED}/pretrain_${stage} --use_ft --n_steps 180 --pe_accuracy 0.002
#     pixi run -e jazzy python crisp_drl/scripts/eval_actor.py --run_name 1cft_d300v9_20_75_s${SEED}/pretrain_${stage} --use_ft --n_steps 180 --pe_accuracy 0.002


# done

# PRETRAIN_STAGES=(1 2 3 4 5 6 7 8 9 10)
# for stage in "${PRETRAIN_STAGES[@]}"; do

#     echo "Stage (2) ${stage}"

#     pixi run -e jazzy python crisp_drl/scripts/eval_actor.py --run_name 1cft_d400v9_15_s${SEED}/pretrain_${stage} --use_ft
#     pixi run -e jazzy python crisp_drl/scripts/eval_actor.py --run_name 1cft_d400v9_20_s${SEED}/pretrain_${stage} --use_ft --n_steps 180 --pe_accuracy 0.002
#     pixi run -e jazzy python crisp_drl/scripts/eval_actor.py --run_name 1cft_d400v9_20_75_s${SEED}/pretrain_${stage} --use_ft --n_steps 180 --pe_accuracy 0.002
#     pixi run -e jazzy python crisp_drl/scripts/eval_actor.py --run_name 1cft_d500v9_20_s${SEED}/pretrain_${stage} --use_ft --n_steps 180 --pe_accuracy 0.002
#     pixi run -e jazzy python crisp_drl/scripts/eval_actor.py --run_name 1cft_d500v9_20_75_s${SEED}/pretrain_${stage} --use_ft --n_steps 180 --pe_accuracy 0.002

# done

# PRETRAIN_STAGES=(1 2 3 4 5 6 7 8)
# for stage in "${PRETRAIN_STAGES[@]}"; do

#     echo "Stage (3) ${stage}"

#     pixi run -e jazzy python crisp_drl/scripts/eval_actor.py --run_name 1cft_d400v9_25_s${SEED}/pretrain_${stage} --use_ft --n_steps 240 --pe_accuracy 0.0025
#     pixi run -e jazzy python crisp_drl/scripts/eval_actor.py --run_name 1cft_d400v9_25_60_s${SEED}/pretrain_${stage} --use_ft --n_steps 240 --pe_accuracy 0.0025
#     pixi run -e jazzy python crisp_drl/scripts/eval_actor.py --run_name 1cft_d500v9_25_s${SEED}/pretrain_${stage} --use_ft --n_steps 240 --pe_accuracy 0.0025
#     pixi run -e jazzy python crisp_drl/scripts/eval_actor.py --run_name 1cft_d500v9_25_60_s${SEED}/pretrain_${stage} --use_ft --n_steps 240 --pe_accuracy 0.0025

#     pixi run -e jazzy python crisp_drl/scripts/eval_actor.py --run_name 1cft_d600v9_25_s${SEED}/pretrain_${stage} --use_ft --n_steps 240 --pe_accuracy 0.0025
#     pixi run -e jazzy python crisp_drl/scripts/eval_actor.py --run_name 1cft_d600v9_25_60_s${SEED}/pretrain_${stage} --use_ft --n_steps 240 --pe_accuracy 0.0025

# done



PRETRAIN_STAGES=(0.10 0.30 0.50 0.70 0.90)
for stage in "${PRETRAIN_STAGES[@]}"; do

    echo "Stage (3) ${stage}"

    pixi run -e jazzy python crisp_drl/scripts/eval_actor.py --run_name 1cft_d500v9_30_s${SEED}/pretrain_${stage} --use_ft --n_steps 330 --pe_accuracy 0.003
    pixi run -e jazzy python crisp_drl/scripts/eval_actor.py --run_name 1cft_d600v9_30_s${SEED}/pretrain_${stage} --use_ft --n_steps 330 --pe_accuracy 0.003
    pixi run -e jazzy python crisp_drl/scripts/eval_actor.py --run_name 1cft_d700v9_30_s${SEED}/pretrain_${stage} --use_ft --n_steps 330 --pe_accuracy 0.003
    pixi run -e jazzy python crisp_drl/scripts/eval_actor.py --run_name 1cft_d800v9_30_s${SEED}/pretrain_${stage} --use_ft --n_steps 330 --pe_accuracy 0.003

done

# PRETRAIN_STAGES=(1 2 3 4 5 6 7 8 9 10)
# for stage in "${PRETRAIN_STAGES[@]}"; do

#     echo "Stage (3) ${stage}"

#     pixi run -e jazzy python crisp_drl/scripts/eval_actor.py --run_name 1cft_d500v9_30_40_s${SEED}/pretrain_${stage} --use_ft --n_steps 330 --pe_accuracy 0.003
#     pixi run -e jazzy python crisp_drl/scripts/eval_actor.py --run_name 1cft_d600v9_30_40_s${SEED}/pretrain_${stage} --use_ft --n_steps 330 --pe_accuracy 0.003
#     pixi run -e jazzy python crisp_drl/scripts/eval_actor.py --run_name 1cft_d700v9_30_40_s${SEED}/pretrain_${stage} --use_ft --n_steps 330 --pe_accuracy 0.003
#     pixi run -e jazzy python crisp_drl/scripts/eval_actor.py --run_name 1cft_d800v9_30_40_s${SEED}/pretrain_${stage} --use_ft --n_steps 330 --pe_accuracy 0.003


# done












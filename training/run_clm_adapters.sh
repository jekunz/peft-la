#!/bin/bash
#SBATCH -A NAISS2024-22-1172 -p alvis
#SBATCH -N 1 --gpus-per-node=A100:2
#SBATCH -t 2-00:00:00


apptainer exec adapters-env.sif python run_clm_adapters.py \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct  \
    --train_file is.txt  \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --max_train_samples 250000 \
    --block_size 1024 \
    --do_train \
    --output_dir /mimer/NOBACKUP/groups/langadapters/adapters/lora_is/lora1024 \
    --train_adapter \
    --token true \
    --adapter_config lora \
    --push_to_hub true 
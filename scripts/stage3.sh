#!/bin/bash

MODEL_VERSION=vicuna-v1-5-7b
gpu_vis=0 # per_device_train_batch_size * gradient_accumulation_steps * n_gpus = 128
MASTER_PORT=29570

cd ..

deepspeed --include localhost:$gpu_vis --master_port $MASTER_PORT vtimellm/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --training_stage 3 \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path ./data/xl/train/results_train.json \
    --feat_folder ./data/xl/train/stage4_features  \
    --val_data_path ./data/xl/val/results_val.json \
    --val_feat_folder ./data/xl/val/stage4_features_val \
    --pretrain_mm_mlp_adapter ./checkpoints/vtimellm-$MODEL_VERSION-stage1/mm_projector.bin \
    --stage2_path ./checkpoints/vtimellm-$MODEL_VERSION-stage2 \
    --output_dir ./checkpoints/vtimellm-$MODEL_VERSION-stage3_xl_300_epoch \
    --bf16 True \
    --num_train_epochs 300 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "steps" \
    --eval_steps 100 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 5 \
    --load_best_model_at_end True \
    --metric_for_best_model "eval_loss" \
    --greater_is_better False \
    --learning_rate 1e-4 \
    --tune_mm_mlp_adapter True \
    --freeze_mm_mlp_adapter False \
    --lora_r 64 \
    --lora_alpha 128 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb

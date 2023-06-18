#!/bin/bash

MODEL_PATH="outputs/archconvnext_lr0.0005_freezeT_optimadamw_VnColor-customCos5e-4__0.4_B32/ckpt/ep196.pt"

for LR in 5e-5
do
    python train.py --lr $LR --arch_ver convnext --ver_name convnext-FINETUNE_$LR"_COS_WD1e-4_LB0_B32" --batch 32 --labelsmooth 0 --optim_type "adamw" --weight_decay 1e-4 --epoch 70 --pretrained $MODEL_PATH
done

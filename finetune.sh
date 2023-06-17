#!/bin/bash

MODEL_PATH="outputs/archver1_lr0.001_freezeT_optimadamw_VnewRes_NocolorT_lr2e-3_B256_WD0/ckpt/ep48.pt"

for LR in 1e-4 5e-5 1e-5
do
    python train.py --lr $LR --ver_name sFINETUNE_$LR"_COS_WD0_LB0" --labelsmooth 0 --optim_type "adamw" --weight_decay 0 --epoch 70 --pretrained $MODEL_PATH
done

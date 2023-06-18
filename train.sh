#!/bin/bash


for LR in 5e-4
do
    for LB in 0 0.1 0.2 0.3 0.4
    do
            python train.py --lr $LR --arch_ver convnext --ver_name wColor-customCos$LR'_'$WD'_'$LB'_B32' --batch 32 --labelsmooth $LB --optim_type "adamw" --weight_decay 1e-4 --epoch 200
    done
done

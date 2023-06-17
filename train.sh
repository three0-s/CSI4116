#!/bin/bash


for LR in 1e-3 5e-3
do
    for LB in 0 0.1
    do
            python train.py --lr $LR --freeze --arch_ver convnext --ver_name dynamic_nColor-customCos$LR'_'$WD'_'$LB --labelsmooth $LB --optim_type "adamw" --weight_decay 1e-4 --epoch 200
    done
done

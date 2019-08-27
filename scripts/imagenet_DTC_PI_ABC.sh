#!/usr/bin/env bash

names='A B C'

for name in $names
do

    python imagenet_DTC.py \
          --DTC PI \
          --subset $name \
          --model_name PI_run_$name \
          --save_txt_name PI_results_ABC.txt \
          --warmup_lr 0.1 \
          --lr 0.05 \
          --warmup_epochs 10 \
          --epochs 60 \
          --batch_size 128 \
          --rampup_length 5 \
          --rampup_coefficient 100.0 \
          --update_interval 5 \
          --weight_decay 1e-5 \
          --save_txt true \

done

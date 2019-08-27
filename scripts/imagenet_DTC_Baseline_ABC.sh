#!/usr/bin/env bash

names='A B C'

for name in $names
do

    python imagenet_DTC.py \
          --DTC Baseline \
          --subset $name \
          --model_name Baseline_run_$name \
          --save_txt_name Baseline_results_ABC.txt \
          --warmup_lr 0.1 \
          --lr 0.05 \
          --warmup_epochs 10 \
          --epochs 60 \
          --batch_size 128 \
          --update_interval 5 \
          --weight_decay 1e-5 \
          --save_txt true \

done

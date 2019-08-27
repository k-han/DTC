#!/usr/bin/env bash

for i in {1..10}
do

    python svhn_DTC.py \
          --DTC Baseline \
          --model_name Baseline_run_$i \
          --save_txt_name Baseline_results_10runs.txt \
          --seed $i \
          --warmup_lr 0.1 \
          --lr 0.001 \
          --warmup_epochs 10 \
          --epochs 60 \
          --batch_size 128 \
          --update_interval 5 \
          --weight_decay 0 \
          --save_txt true \

done

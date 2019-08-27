#!/usr/bin/env bash

for i in {1..10}
do

    python cifar100_DTC.py \
          --DTC Baseline \
          --model_name Baseline_run_$i \
          --save_txt_name Baseline_results_10runs.txt \
          --seed $i \
          --warmup_lr 0.1 \
          --lr 0.05 \
          --warmup_epochs 30 \
          --epochs 100 \
          --batch_size 128 \
          --gamma 0.5 \
          --update_interval 10 \
          --weight_decay 1e-5 \
          --milestones 20 40 60 80 \
          --save_txt true \

done

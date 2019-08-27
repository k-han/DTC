#!/usr/bin/env bash

for i in {1..10}
do

    python svhn_DTC.py \
          --DTC PI \
          --model_name PI_run_$i \
          --save_txt_name PI_results_10runs.txt \
          --seed $i \
          --warmup_lr 0.1 \
          --lr 0.001 \
          --warmup_epochs 10 \
          --epochs 60 \
          --batch_size 128 \
          --rampup_length 5 \
          --rampup_coefficient 10.0 \
          --update_interval 5 \
          --weight_decay 0 \
          --save_txt true \

done

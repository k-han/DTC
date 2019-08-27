#!/usr/bin/env bash

for i in {1..10}
do

    python cifar10_DTC.py \
          --DTC TEP \
          --model_name TEP_run_$i \
          --save_txt_name TEP_results_10runs.txt \
          --seed $i \
          --warmup_lr 0.1 \
          --lr 0.05 \
          --warmup_epochs 10 \
          --epochs 100 \
          --batch_size 128 \
          --update_interval 5 \
          --weight_decay 1e-5 \
          --save_txt true \

done

#!/usr/bin/env bash

for i in {1..10}
do

    python omniglot_DTC.py \
          --DTC PI \
          --subfolder_name run_$i \
          --save_txt_name PI_results_10runs.txt \
          --seed $i \
          --num_workers 2 \
          --warmup_lr 0.001 \
          --lr 0.001 \
          --warmup_epochs 10 \
          --epochs 90 \
          --batch_size 100 \
          --rampup_length 5 \
          --rampup_coefficient 100.0 \
          --save_txt true \

done

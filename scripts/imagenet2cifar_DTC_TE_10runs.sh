#!/usr/bin/env bash

for i in {1..10}
do

    python imagenet2cifar_DTC.py \
          --DTC TE \
          --model_name TE_run_$i \
          --save_txt_name TE_results_10runs.txt \
          --seed $i \
          --warmup_lr 0.1 \
          --lr 0.1 \
          --warmup_epochs 10 \
          --epochs 100 \
          --batch_size 128 \
          --rampup_length 5 \
          --rampup_coefficient 10.0 \
          --gamma 0.5 \
          --update_interval 5 \
          --weight_decay 1e-5 \
          --milestones 20 40 60 80 \
          --save_txt true \

done

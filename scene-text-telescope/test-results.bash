#!/bin/bash

for filename in checkpoint/$1/epoch*_.pth; do
    echo "=============================="
    echo $filename
    bash -c "WANDB_MODE=disabled python3 main.py --resume $filename --text_focus --STN --test --batch_size=16 --exp_name $1 --test_data_dir ./dataset/mydata/test | tee log.txt"
    echo ""
    echo ""
done

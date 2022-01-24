#!/bin/bash

for filename in checkpoint/$1/epoch*_.pth; do
    echo "=============================="
    echo $filename
    bash -c "python3 main.py --resume $filename --text_zoom --STN --test --batch_size=16 --exp_name $1 --test_data_dir ./dataset/mydata/test"
    echo ""
    echo ""
done

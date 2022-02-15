#!/bin/sh
# Test CAR with cDist
# python3 main.py --batch_size=16 --STN --exp_name '' --text_focus --resume checkpoint/stnworkingwtfGradients/epoch5_.pth --cdistresume /home/philipp/FudanOCR/scene-text-telescope/checkpoint/carcdistfinetunedsave/epoch7_.pth --rec cdist --test --test_data_dir ./dataset/mydata/test --arch car

# Test TBSRN with cDist
# python3 main.py --batch_size=16 --STN --exp_name '' --text_focus --resume model_best.pth --cdistresume /home/philipp/FudanOCR/scene-text-telescope/checkpoint/finetundcdisttbsrn/epoch3_.pth --rec cdist --test --test_data_dir ./dataset/mydata/test

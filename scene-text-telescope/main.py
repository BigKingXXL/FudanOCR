import os
import yaml
import argparse
from IPython import embed
from easydict import EasyDict
from interfaces.super_resolution import TextSR

def main(config, args):
    Mission = TextSR(config, args)
    if args.test:
        for epoch in range(3):
            Mission.test(quantize_static=args.quantize_static, modeldir=f"/home/philipp/FudanOCR/scene-text-telescope/checkpoint/CARcdist/epoch{epoch}_.pth", bitsize=8)
    elif args.demo:
        Mission.demo()
    elif args.train_cdist:
        Mission.train_rec()
    else:
        Mission.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--arch', default='tbsrn', choices=['tbsrn', 'tsrn', 'bicubic', 'srcnn', 'vdsr', 'srres', 'esrgan', 'rdn',
                                                           'edsr', 'lapsrn'])
    parser.add_argument('--text_focus', action='store_true')
    parser.add_argument('--exp_name', required=True, help='Type your experiment name')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--test_data_dir', type=str, default='./dataset/mydata/test')
    parser.add_argument('--batch_size', type=int, default=None, help='')
    parser.add_argument('--resume', type=str, default='', help='')
    parser.add_argument('--rec', default='crnn', choices=['crnn', 'cdist'])
    parser.add_argument('--STN', action='store_true', default=False, help='')
    parser.add_argument('--syn', action='store_true', default=False, help='use synthetic LR')
    parser.add_argument('--mixed', action='store_true', default=False, help='mix synthetic with real LR')
    parser.add_argument('--mask', action='store_true', default=False, help='')
    parser.add_argument('--hd_u', type=int, default=32, help='')
    parser.add_argument('--srb', type=int, default=5, help='')
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--demo_dir', type=str, default='./demo')
    parser.add_argument('--quantize', action='store_true')
    parser.add_argument('--quantize_static', action='store_true')
    parser.add_argument('--train_cdist', action='store_true')
    args = parser.parse_args()
    config_path = os.path.join('config', 'super_resolution.yaml')
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    config = EasyDict(config)
    main(config, args)

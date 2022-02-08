import codecs
import os
import sys
import torch
import shutil
import string
import logging
import torchvision
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
from EDSR.edsr import EDSR
from model import tbsrn, tsrn, edsr, srcnn, srresnet, crnn
import dataset.dataset as dataset
from dataset import lmdbDataset, alignCollate_real, lmdbDataset_real, alignCollate_syn, lmdbDataset_mix
from loss import text_focus_loss
from model.cdistnet.cdistnet.data.data import src_pad
from model.cdistnet.cdistnet.model.model import CDistNet
from model.cdistnet.cdistnet.model.translator import Translator
from model.qtbsrn import QTBSRN
from utils import ssim_psnr, utils_moran, utils_crnn, utils_cdist
from utils.labelmaps import get_vocabulary

from interfaces.aciq import *
from interfaces.q_utils import *
from interfaces.ocs import *
from interfaces.clip import *
import model.cdistnet.build as cdistnet_build


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    return total_num

class MyEnsemble(torch.nn.Module):
    def __init__(self, modelA, modelB):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB

        
    def forward(self, x):
        #return self.modelB(x)
        return self.modelB(self.modelA(x))

class TextBase(object):
    def __init__(self, config, args):
        super(TextBase, self).__init__()
        self.config = config
        self.args = args
        self.scale_factor = self.config.TRAIN.down_sample_scale
        if self.args.syn:
            self.align_collate = alignCollate_syn
            self.load_dataset = lmdbDataset
        elif self.args.mixed:
            self.align_collate = alignCollate_real
            self.load_dataset = lmdbDataset_mix
        else:
            self.align_collate = alignCollate_real
            self.load_dataset = lmdbDataset_real
        self.resume = args.resume if args.resume is not None else config.TRAIN.resume
        self.batch_size = args.batch_size if args.batch_size is not None else self.config.TRAIN.batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        alpha_dict = {
            'digit': string.digits,
            'lower': string.digits + string.ascii_lowercase,
            'upper': string.digits + string.ascii_letters,
            'all': string.digits + string.ascii_letters + string.punctuation
        }
        self.test_data_dir = self.args.test_data_dir if self.args.test_data_dir is not None else self.config.TEST.test_data_dir
        self.voc_type = self.config.TRAIN.voc_type
        self.alphabet = alpha_dict[self.voc_type]
        self.max_len = config.TRAIN.max_len
        # self.vis_dir = self.args.vis_dir if self.args.vis_dir is not None else self.config.TRAIN.VAL.vis_dir
        self.cal_psnr = ssim_psnr.calculate_psnr
        self.cal_ssim = ssim_psnr.SSIM()
        self.mask = self.args.mask
        alphabet_moran = ':'.join(string.digits+string.ascii_lowercase+'$')
        self.converter_moran = utils_moran.strLabelConverterForAttention(alphabet_moran, ':')
        self.converter_crnn = utils_crnn.strLabelConverter(string.digits + string.ascii_lowercase)
        self.converter_cdist = utils_cdist.strLabelConverter(['<blank>', '<unk>', '<s>', '</s>', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])
        if not args.test and not args.demo:
            self.clean_old_ckpt()
        self.logging = logging
        self.make_logger()
        self.make_writer()

    def make_logger(self):
        self.logging.basicConfig(filename="checkpoint/{}/log.txt".format(self.args.exp_name),
                            level=self.logging.INFO,
                            format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        self.logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        self.logging.info(str(self.args))

    def clean_old_ckpt(self):
        if os.path.exists('checkpoint/{}'.format(self.args.exp_name)):
            shutil.rmtree('checkpoint/{}'.format(self.args.exp_name))
            print(f'Clean the old checkpoint {self.args.exp_name}')
        os.mkdir('checkpoint/{}'.format(self.args.exp_name))


    def make_writer(self):
        self.writer = SummaryWriter('checkpoint/{}'.format(self.args.exp_name))


    def get_train_data(self):
        cfg = self.config.TRAIN
        if isinstance(cfg.train_data_dir, list):
            dataset_list = []
            for data_dir_ in cfg.train_data_dir:
                dataset_list.append(
                    self.load_dataset(root=data_dir_,
                                      voc_type=cfg.voc_type,
                                      max_len=cfg.max_len))
            train_dataset = dataset.ConcatDataset(dataset_list)
        else:
            raise TypeError('check trainRoot')

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=int(cfg.workers),
            collate_fn=self.align_collate(imgH=cfg.height, imgW=cfg.width, down_sample_scale=cfg.down_sample_scale,
                                          mask=self.mask),
            drop_last=True)
        return train_dataset, train_loader

    def get_val_data(self):
        cfg = self.config.TRAIN
        assert isinstance(cfg.VAL.val_data_dir, list)
        dataset_list = []
        loader_list = []
        for data_dir_ in cfg.VAL.val_data_dir:
            val_dataset, val_loader = self.get_test_data(data_dir_)
            dataset_list.append(val_dataset)
            loader_list.append(val_loader)
        return dataset_list, loader_list

    def get_test_data(self, dir_):
        cfg = self.config.TRAIN
        test_dataset = self.load_dataset(root=dir_,
                                         voc_type=cfg.voc_type,
                                         max_len=cfg.max_len,
                                         test=True,
                                         )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=int(cfg.workers),
            collate_fn=self.align_collate(imgH=cfg.height, imgW=cfg.width, down_sample_scale=cfg.down_sample_scale,
                                          mask=self.mask),
            drop_last=False)
        return test_dataset, test_loader

    def generator_init(self, small=False, quantized=False, quantize_static=False):
        cfg = self.config.TRAIN
        if self.args.arch == 'tbsrn':
            if quantized:
                model = QTBSRN(scale_factor=self.scale_factor, width=cfg.width, height=cfg.height,
                                STN=self.args.STN, mask=self.mask, srb_nums=self.args.srb, hidden_units=self.args.hd_u)
            else:
               model = tbsrn.TBSRN(scale_factor=self.scale_factor, width=cfg.width, height=cfg.height,
                                STN=self.args.STN, mask=self.mask, srb_nums=self.args.srb, hidden_units=self.args.hd_u, small=small, quantize_static=quantize_static)
            if self.args.rec == "cdist":
                _, cdist_model = self.cdistnet_init(self.args.cdistresume)
                cdist_model.eval()
                image_crit = text_focus_loss.TextFocusLoss(self.args, recognition_model=cdist_model)
            else:
                image_crit = text_focus_loss.TextFocusLoss(self.args)
        elif self.args.arch == 'car':
            SCALE=2
            edsr_model = EDSR(32, 256, scale=SCALE).cuda()
            stn_model = tbsrn.STNmodel(scale_factor=self.scale_factor, width=cfg.width, height=cfg.height,
                            STN=self.args.STN, mask=self.mask, srb_nums=self.args.srb, hidden_units=self.args.hd_u, small=small, quantize_static=quantize_static)
            model = MyEnsemble(stn_model, edsr_model)
            if self.args.rec == "cdist":
                _, cdist_model = self.cdistnet_init(self.args.cdistresume)
                cdist_model.eval()
                image_crit = text_focus_loss.TextFocusLoss(self.args, recognition_model=cdist_model)
            else:
                image_crit = text_focus_loss.TextFocusLoss(self.args)
        elif self.args.arch == 'tsrn':
            model = tsrn.TSRN(scale_factor=self.scale_factor, width=cfg.width, height=cfg.height,
                              STN=self.args.STN, mask=self.mask, srb_nums=self.args.srb, hidden_units=self.args.hd_u)
            image_crit = text_focus_loss.TextFocusLoss(self.args)
        elif self.args.arch == 'bicubic' and self.args.test:
            model = bicubic.BICUBIC(scale_factor=self.scale_factor)
            image_crit = text_focus_loss.TextFocusLoss(self.args)
        elif self.args.arch == 'srcnn':
            model = srcnn.SRCNN(scale_factor=self.scale_factor, width=cfg.width, height=cfg.height, STN=self.args.STN)
            image_crit = text_focus_loss.TextFocusLoss(self.args)
        elif self.args.arch == 'vdsr':
            model = vdsr.VDSR(scale_factor=self.scale_factor, width=cfg.width, height=cfg.height, STN=self.args.STN)
            image_crit = nn.MSELoss()
        elif self.args.arch == 'srres':
            model = srresnet.SRResNet(scale_factor=self.scale_factor, width=cfg.width, height=cfg.height,
                                      STN=self.args.STN, mask=self.mask)
            image_crit = text_focus_loss.TextFocusLoss()
        elif self.args.arch == 'esrgan':
            model = esrgan.RRDBNet(scale_factor=self.scale_factor)
            image_crit = nn.L1Loss()
        elif self.args.arch == 'rdn':
            model = rdn.RDN(scale_factor=self.scale_factor)
            image_crit = nn.L1Loss()
        elif self.args.arch == 'edsr':
            model = edsr.EDSR(scale_factor=self.scale_factor)
            image_crit = nn.L1Loss()
        elif self.args.arch == 'lapsrn':
            model = lapsrn.LapSRN(scale_factor=self.scale_factor, width=cfg.width, height=cfg.height, STN=self.args.STN)
            image_crit = lapsrn.L1_Charbonnier_loss()
        else:
            raise ValueError
        print("Resume:")
        print(self.resume)
        if self.args.arch != 'bicubic':
            model = model.to(self.device)
            image_crit.to(self.device)
            if cfg.ngpu > 1:
                model = torch.nn.DataParallel(model)
                # image_crit = torch.nn.DataParallel(image_crit)
            if self.resume is not '':
                self.logging.info('loading pre-trained model from %s ' % self.resume)
                if self.args.arch == 'car':
                    weights = {}
                    for k,v in torch.load(self.resume)['state_dict_G'].items():
                        if k.startswith("modelA.module"):
                            weights["module.modelA."+k[14:]] = v
                        elif k.startswith("modelB"):
                            weights["module."+k] = v
                        else:
                            weights[k] = v
                    model.load_state_dict(weights)
                else:
                    if self.config.TRAIN.ngpu == 1:
                        weights = torch.load(self.resume)['state_dict_G']
                        # if quantized:
                        #     for key in ["block2.conv1.bias", "block2.conv2.bias", "block3.conv1.bias", "block3.conv2.bias", "block4.conv1.bias", "block4.conv2.bias", "block5.conv1.bias", "block5.conv2.bias", "block6.conv1.bias", "block6.conv2.bias"]:
                        #         del weights[key]
                        #model.load_state_dict(weights)
                    else:
                        weights = {'module.' + k: v  for k, v in torch.load(self.resume)['state_dict_G'].items()}# if ('tps' in k or 'stn' in k)}
                        #print(weights)
                        # if quantized:
                        #     for key in ["module.block2.conv1.bias", "module.block2.conv2.bias", "module.block3.conv1.bias", "module.block3.conv2.bias", "module.block4.conv1.bias", "module.block4.conv2.bias", "module.block5.conv1.bias", "module.block5.conv2.bias", "module.block6.conv1.bias", "module.block6.conv2.bias"]:
                        #         del weights[key]
                    model.load_state_dict(weights)

                bits = 5
                uncompressed_size = 0
                compressed_size = 0
                all_w_count = 0
                qat_w_count = 0
                qat_method = 'aciq'

                method = 'none'

                if method == 'aciq':
                    for key, value in model.named_parameters():
                        if 'gru' not in key:
                            uncompressed_size += value.data.element_size() * 8 * value.data.numel()
                        all_w_count += value.data.numel()
                        #print(key)
                        quantization_keys = [
                            'conv',
                            'multihead',
                            'linear',
                            '.pff.',
                            'stn_fc'
                        ]
                        if (any(name in key for name in quantization_keys)):
                            #print('compressing ' + key + ' ' + str(value.shape) + ' to ' + str(bits) + 'bits using ' + qat_method)
                            weight_np = value.data.cpu().detach().numpy()
                            qat_w_count += value.data.numel()
                            # print(weight_np)
                            # obtain value range
                            params_min_q_val, params_max_q_val = get_quantized_range(bits, signed=True)
                            # find clip threshold
                            if qat_method == 'lq':  # fix threshold
                                clip_max_abs = np.max(np.abs(weight_np))
                            elif qat_method == 'aciq':  # calculate threshold
                                values = weight_np.flatten().copy()
                                clip_max_abs = find_clip_aciq(values, bits)
                
                            # quantize weights
                            w_scale = symmetric_linear_quantization_scale_factor(bits, clip_max_abs)
                            q_weight_np = linear_quantize_clamp(weight_np, w_scale, params_min_q_val, params_max_q_val, inplace=False)
                
                            # dequantize/rescale
                            q_weight_np = linear_dequantize(q_weight_np, w_scale)
                            # print(q_weight_np)
                
                            # update weight
                            value.data = torch.tensor(q_weight_np).to(self.device)
                            if 'gru' not in key:
                                compressed_size += bits * value.data.numel()
                        else:
                
                            if 'gru' not in key:
                                print(key)
                                compressed_size += value.data.element_size() * 8 * value.data.numel()
                if method == 'ocs':
                    #implementation form https://github.com/cornell-zhang/dnn-quant-ocs/blob/ca3a413c73850b4e5d7aac558f5856b44060e39c/distiller/quantization/ocs.py
                    weight_expand_ratio=0.0
                    weight_clip_threshold=1.0   
                    split_threshold = 0.0

                    for key, value in model.named_parameters():
                        if 'gru' not in key:
                            uncompressed_size += value.data.element_size() * 8 * value.data.numel()
                        all_w_count += value.data.numel()
                        #print(key)
                        quantization_keys = [
                            'conv',
                            'multihead',
                            'linear',
                            '.pff.',
                            'stn_fc'
                        ]
                        if (any(name in key for name in quantization_keys)):
                            weight_np = value.data.cpu().detach().numpy()
                            qat_w_count += value.data.numel()

                            # Perform prelim OCS
                            q_weight_np, in_channels_to_split = ocs_wts(
                                    weight_np,
                                    weight_expand_ratio,
                                    split_threshold=split_threshold,
                                    grid_aware=False)

                        # Find the clip threshold (alpha value in clipping papers)
                        if weight_clip_threshold > 0.0:
                            # Fixed threshold
                            max_abs = get_tensor_max_abs(q_weight_np)
                            clip_max_abs = weight_clip_threshold * max_abs
                        else:
                            print('Auto-tuning for weight clip threshold...')
                            # Calculate threshold
                            values = weight_np.flatten().copy()

                            # Branch on clip method
                            if weight_clip_threshold == 0.0:
                                clip_max_abs = find_clip_mmse(values, bits)
                            elif self.weight_clip_threshold == -1.0:
                                clip_max_abs = find_clip_aciq(values, bits)
                            elif self.weight_clip_threshold == -2.0:
                                clip_max_abs = find_clip_entropy(values, bits)
                            else:
                                raise ValueError('Undefined weight clip method')

                        w_scale = symmetric_linear_quantization_scale_factor(bits, clip_max_abs)

                        # Grid aware OCS
                        q_weight_np, in_channels_to_split = ocs_wts(
                                weight_np,
                                weight_expand_ratio,
                                split_threshold=split_threshold,
                                w_scale=w_scale,
                                grid_aware=True)

                        q_weight_np = linear_quantize_clamp(q_weight_np, w_scale, params_min_q_val, params_max_q_val, inplace=False)
                
                        # dequantize/rescale
                        q_weight_np = linear_dequantize(q_weight_np, w_scale)
                
                        # recombine channels
                        if weight_np.ndim > 1:
                            recons_weight_np = q_weight_np[:, :weight_np.shape[1]].copy()
                            for k in range(len(in_channels_to_split)):
                                recons_weight_np[:, in_channels_to_split[k]:in_channels_to_split[k]+1] += \
                                    q_weight_np[:, weight_np.shape[1]+k: weight_np.shape[1]+k+1]
                        else:
                            recons_weight_np = q_weight_np[:weight_np.shape[0]].copy()
                            for k in range(len(in_channels_to_split)):
                                recons_weight_np[in_channels_to_split[k]:in_channels_to_split[k] + 1] += \
                                    q_weight_np[weight_np.shape[0] + k: weight_np.shape[0] + k + 1]
                
                        # print(recons_weight_np.shape)
                        # update weight
                        value.data = torch.tensor(recons_weight_np).to(self.device)
                        compressed_size += bits * value.data.numel()
                    else:
                        compressed_size += value.data.element_size() * 8 * value.data.numel()


                print("Compressed size:", compressed_size)
                print("Uncompressed size:", uncompressed_size)
                
        para_num = get_parameter_number(model)

        para_num = get_parameter_number(model)
        self.logging.info('Total Parameters {}'.format(para_num))

        return {'model': model, 'crit': image_crit}

    def optimizer_init(self, model):
        cfg = self.config.TRAIN
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr,
                               betas=(cfg.beta1, 0.999))
        return optimizer

    def tripple_display(self, image_in, image_out, image_target, pred_str_lr, pred_str_sr, label_strs, index):
        for i in (range(self.config.TRAIN.VAL.n_vis)):
            tensor_in = image_in[i][:3,:,:]
            transform = transforms.Compose(
                [transforms.ToPILImage(),
                 transforms.Resize((image_target.shape[-2], image_target.shape[-1]), interpolation=Image.BICUBIC),
                 transforms.ToTensor()]
            )

            tensor_in = transform(tensor_in.cpu())
            tensor_out = image_out[i][:3,:,:]
            tensor_target = image_target[i][:3,:,:]
            images = ([tensor_in, tensor_out.cpu(), tensor_target.cpu()])
            vis_im = torch.stack(images)
            vis_im = torchvision.utils.make_grid(vis_im, nrow=1, padding=0)
            out_root = os.path.join('./demo', self.vis_dir)
            if not os.path.exists(out_root):
                os.mkdir(out_root)
            out_path = os.path.join(out_root, str(index))
            if not os.path.exists(out_path):
                os.mkdir(out_path)
            im_name = pred_str_lr[i] + '_' + pred_str_sr[i] + '_' + label_strs[i] + '_.png'
            im_name = im_name.replace('/', '')
            if index is not 0:
                torchvision.utils.save_image(vis_im, os.path.join(out_path, im_name), padding=0)

    def test_display(self, image_in, image_out, image_target, pred_str_lr, pred_str_sr, label_strs, str_filt):
        visualized = 0
        for i in (range(image_in.shape[0])):
            if True:
                if (str_filt(pred_str_lr[i], 'lower') != str_filt(label_strs[i], 'lower')) and \
                        (str_filt(pred_str_sr[i], 'lower') == str_filt(label_strs[i], 'lower')):
                    visualized += 1
                    tensor_in = image_in[i].cpu()
                    tensor_out = image_out[i].cpu()
                    tensor_target = image_target[i].cpu()
                    transform = transforms.Compose(
                        [transforms.ToPILImage(),
                         transforms.Resize((image_target.shape[-2], image_target.shape[-1]), interpolation=Image.BICUBIC),
                         transforms.ToTensor()]
                    )
                    tensor_in = transform(tensor_in)
                    images = ([tensor_in, tensor_out, tensor_target])
                    vis_im = torch.stack(images)
                    vis_im = torchvision.utils.make_grid(vis_im, nrow=1, padding=0)
                    out_root = os.path.join('./display', self.vis_dir)
                    if not os.path.exists(out_root):
                        os.mkdir(out_root)
                    if not os.path.exists(out_root):
                        os.mkdir(out_root)
                    im_name = pred_str_lr[i] + '_' + pred_str_sr[i] + '_' + label_strs[i] + '_.png'
                    im_name = im_name.replace('/', '')
                    torchvision.utils.save_image(vis_im, os.path.join(out_root, im_name), padding=0)
        return visualized

    def save_checkpoint(self, netG, epoch, iters, best_acc_dict, best_model_info, is_best, converge_list, exp_name, epoch_save=False):
        # ckpt_path = os.path.join('checkpoint', exp_name, self.vis_dir)
        ckpt_path = os.path.join('checkpoint', exp_name)
        if not os.path.exists(ckpt_path):
            os.mkdir(ckpt_path)
        save_dict = {
            'state_dict_G': netG.state_dict(),
            'info': {'arch': self.args.arch, 'iters': iters, 'epochs': epoch, 'batch_size': self.batch_size,
                     'voc_type': self.voc_type, 'up_scale_factor': self.scale_factor},
            'best_history_res': best_acc_dict,
            'best_model_info': best_model_info,
            'param_num': sum([param.nelement() for param in netG.parameters()]),
            'converge': converge_list
        }
        save_dict = netG.state_dict()
        if epoch_save:
            torch.save(netG, os.path.join(ckpt_path, f'epoch{epoch}_.pth'))
        if is_best:
            torch.save(netG, os.path.join(ckpt_path, 'model_best.pth'))
        else:
            torch.save(netG, os.path.join(ckpt_path, 'checkpoint.pth'))

    def MORAN_init(self):
        cfg = self.config.TRAIN
        alphabet = ':'.join(string.digits+string.ascii_lowercase+'$')
        MORAN = moran.MORAN(1, len(alphabet.split(':')), 256, 32, 100, BidirDecoder=True,
                            inputDataType='torch.cuda.FloatTensor', CUDA=True)
        model_path = self.config.TRAIN.VAL.moran_pretrained
        self.logging.info('loading pre-trained moran model from %s' % model_path)
        state_dict = torch.load(model_path)
        MORAN_state_dict_rename = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "")  # remove `module.`
            MORAN_state_dict_rename[name] = v
        MORAN.load_state_dict(MORAN_state_dict_rename)
        MORAN = MORAN.to(self.device)
        MORAN = torch.nn.DataParallel(MORAN, device_ids=range(cfg.ngpu))
        for p in MORAN.parameters():
            p.requires_grad = False
        MORAN.eval()
        return MORAN

    def parse_moran_data(self, imgs_input):
        batch_size = imgs_input.shape[0]
        imgs_input = torch.nn.functional.interpolate(imgs_input, (32, 100), mode='bicubic')
        R = imgs_input[:, 0:1, :, :]
        G = imgs_input[:, 1:2, :, :]
        B = imgs_input[:, 2:3, :, :]
        tensor = 0.299 * R + 0.587 * G + 0.114 * B
        text = torch.LongTensor(batch_size * 5)
        length = torch.IntTensor(batch_size)
        max_iter = 20
        t, l = self.converter_moran.encode(['0' * max_iter] * batch_size)
        utils_moran.loadData(text, t)
        utils_moran.loadData(length, l)
        return tensor, length, text, text

    def CRNN_init(self):
        model = crnn.CRNN(32, 1, 37, 256)
        model = model.to(self.device)
        cfg = self.config.TRAIN
        aster_info = AsterInfo(cfg.voc_type)
        model_path = self.config.TRAIN.VAL.crnn_pretrained
        self.logging.info('loading pretrained crnn model from %s' % model_path)
        model.load_state_dict(torch.load(model_path))
        return model, aster_info
    
    def cdistnet_init(self, path="dataset/10_best_acc.pth"):
        cfg = self.config.TRAIN
        model = cdistnet_build.build_cdistnet(cfg.cdistnet)
        model = model.to(self.device)
        try:
            model = torch.load(path)
        except:
            model.load_state_dict(torch.load(path))
        translator = Translator(cfg.cdistnet, model)
        return translator, model


    def parse_crnn_data(self, imgs_input):
        imgs_input = torch.nn.functional.interpolate(imgs_input, (32, 100), mode='bicubic')
        R = imgs_input[:, 0:1, :, :]
        G = imgs_input[:, 1:2, :, :]
        B = imgs_input[:, 2:3, :, :]
        tensor = 0.299 * R + 0.587 * G + 0.114 * B
        return tensor

    def parse_cdist_data(self, imgs_input):
        if imgs_input.size()[2] == 32 and imgs_input.size()[3] == 128:
            return imgs_input
        imgs_input = torch.nn.functional.interpolate(imgs_input, (32, 128), mode='bicubic')
        tensor = imgs_input / 128. - 1.
        return src_pad(tensor.cpu().numpy()).type_as(imgs_input)


    def Aster_init(self):
        cfg = self.config.TRAIN
        aster_info = AsterInfo(cfg.voc_type)
        aster = recognizer.RecognizerBuilder(arch='ResNet_ASTER', rec_num_classes=aster_info.rec_num_classes,
                                             sDim=512, attDim=512, max_len_labels=aster_info.max_len,
                                             eos=aster_info.char2id[aster_info.EOS], STN_ON=True)
        aster.load_state_dict(torch.load(self.config.TRAIN.VAL.rec_pretrained)['state_dict'])
        self.logging.info('load pred_trained aster model from %s' % self.config.TRAIN.VAL.rec_pretrained)
        aster = aster.to(self.device)
        aster = torch.nn.DataParallel(aster, device_ids=range(cfg.ngpu))
        return aster, aster_info

    def parse_aster_data(self, imgs_input):
        cfg = self.config.TRAIN
        aster_info = AsterInfo(cfg.voc_type)
        input_dict = {}
        images_input = imgs_input.to(self.device)
        input_dict['images'] = images_input * 2 - 1
        batch_size = images_input.shape[0]
        input_dict['rec_targets'] = torch.IntTensor(batch_size, aster_info.max_len).fill_(1)
        input_dict['rec_lengths'] = [aster_info.max_len] * batch_size
        return input_dict


class AsterInfo(object):
    def __init__(self, voc_type):
        super(AsterInfo, self).__init__()
        self.voc_type = voc_type
        assert voc_type in ['digit', 'lower', 'upper', 'all']
        self.EOS = 'EOS'
        self.max_len = 100
        self.PADDING = 'PADDING'
        self.UNKNOWN = 'UNKNOWN'
        self.voc = get_vocabulary(voc_type, EOS=self.EOS, PADDING=self.PADDING, UNKNOWN=self.UNKNOWN)
        self.char2id = dict(zip(self.voc, range(len(self.voc))))
        self.id2char = dict(zip(range(len(self.voc)), self.voc))
        self.rec_num_classes = len(self.voc)

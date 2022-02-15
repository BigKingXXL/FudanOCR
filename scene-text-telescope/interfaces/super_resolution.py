from cProfile import label
import os
import time
import copy
import torch
import random
import logging
import numpy as np
from PIL import Image
from tqdm import tqdm
from IPython import embed
from interfaces import base
from datetime import datetime
from utils.util import str_filt
from torchvision import transforms
from utils.metrics import get_str_list
import wandb
from torch.utils.data import DataLoader
import model.cdistnet.cdistnet.optim.loss as cdist_loss
import model.cdistnet.cdistnet.data.data as cdist_data
import cv2
from EDSR.edsr import EDSR
import os
import wandb
from interfaces.aciq import *
from interfaces.q_utils import *
from interfaces.ocs import *
from interfaces.clip import *
from model.cdistnet.cdistnet.optim.loss import cal_performance
import torch.optim as optim
from octotorch import OctoTorch


wandb.init(project="BigKingXXL", entity="bigkingxxl", save_code=True)
#os.environ['WANDB_MODE'] = 'offline'
to_pil = transforms.ToPILImage()

times = 0
easy_test_times = 0
medium_test_times = 0
hard_test_times = 0

SCALE = 2
KSIZE = 3 * SCALE + 1
OFFSET_UNIT = SCALE
USNPATH = '/home/philipp/FudanOCR/scene-text-telescope/2x/usn.pth'
MODELDIR = '/home/philipp/FudanOCR/scene-text-telescope/checkpoint/stnworkingwtfGradients/epoch5_.pth'
ADVANTAGE_STRINGS = ['11:00am-11:00', '11:00am-11:00', '11:00am-11:00', '11:00am-11:00', 'Quickly', 'Brown', 'Quickly', '100%', 'Mochi', 'in', '20', '92', '30%', 'paper', 'Products', '20', 'paper', '20', '92', '30%', 'copy', 'paper', '20', '20', '92', '92', 'copy', 'Products', '92', 'in', 'Products', 'Ch', 'GUINEA', 'FOOD', 'GUINEA', 'Pure', 'THE', 'ARTS', 'BOY', 'ANDY', 'LEXUS', 'LEXUS', 'NO', 'PARKING', 'NO', 'PARKING', '3', 'PARKING', 'PM', 'PM', '3', '3', 'PARKING', 'PARKING', '900', 'WAY', 'WAY', '900', 'BAY', 'SHUTTLE,', 'OAK,', '510-401-4657', '510-401-4657', '28476-P', 'CHICKEN', 'CHICKEN', 'of', '1', 'Thickness', 'ES', 'TRADER', 'Nuts', 'Raw', 'Raw', 'CHAPMAN', 'CHAPMAN', 'UNIVERSITY', 'EST.', 'PARKING', 'PARKING', 'TRESPASSING', 'TRESPASSING', 'VEHICLES', 'VEHICLES', '644-6744', '644-6744', 'SITE', 'SITE', 'DE', 'DE', 'DE', 'DE', 'NO', 'NO', 'ALCOHOLIC', 'ALCOHOLIC', 'BEVERAGES', 'NO', 'NO', 'NO', 'NO', 'LOUD', 'LOUD', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'WEAPONS', 'NO', 'NO', 'SMOKING', 'NO', 'NO', 'EXCEPTIONS', 'EXCEPTIONS', 'MATERIALS', 'DAILY', 'NO', 'NO', 'BEBIDAS', 'NO', 'NO', 'ANIMALES', 'NO', 'NO', 'MUSICAALTA', 'AUDIFONOS', 'NO', 'NO', 'DROGAS', 'DROGAS', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'FUMAR', 'NO', 'NO', 'iCALIDAD', 'MATERIALES', 'DE', 'DESECHO', 'MANDATORY', 'ON', 'ON', 'SITE', 'OBLIGATORIOS', 'OBLIGATORIOS', 'ENFORZADA', 'AvalonBay', 'of', 'this', 'Market', 'P', 'P', 'A', 'A', 'during', 'UPS', 'GROWTH', 'IN', 'AN', 'ERA', 'OF', 'THE', 'BREAKTHROUGH', 'IDEAS', 'IDEAS', 'A', 'AN', 'ERA', 'SMALL', 'SMALL', 'DISCOVERIES', 'THE', 'READY,SET,DOMINATE', 'THE', 'THE', 'a', 'a', 'Strategic', 'a', 'Comp', 'Comp', 'CROSS', 'THE', 'NEW', 'THE', 'OF', 'of', 'P', '62nd', '62nd', 'St', 'WAY', 'NO', 'SUPER', 'SUPER', 'ODYSSEY', 'TROPICAL', 'MARIOTENNIS', 'ADAMS', 'NOW', '(408)', 'Christie', 'NO', 'PARKING', 'PARKING', 'NO', 'NO', 'PARKING', 'PARKING', 'SWEEPING', 'Christie', 'Christie', 'NO', 'COMMERCIAL', 'PARKING', 'P', 'St', 'Market', 'Dr', 'Nintendo.', 'R', 'of', 'NO', 'RTENCIA:', 'Honey', 'Bucket', 'JUL', 'ON', 'THORNE', 'THE', 'THE', 'SECOND', 'IDEAS', 'IDEAS', 'Performances', 'for', 'Film', '3', '2', '2', 'THE', 'John', 'Stoner', 'THE', 'Dea', 'Dea', 'THE', 'THE', 'IN', 'IN', 'author', 'bestselling', 'THE', 'OF', 'OF', 'of', 'a', 'ARTHUR', 'ARTHUR', 'Rogers', 'Mindfulness', 'Mindfulness', 'Law', 'MNOOKIN', 'TULUMELLO', 'Mindful', 'PARKING', 'LENBROOK', 'BASIC', 'FORTRAN', 'FORTRAN', '3', 'PASCAL', 'NEANDERTHAL', 'ASSEMBLER', 'ASSEMBLER', 'NONSENSE', 'PIGLATIN', '8000', 'THE', 'THE', '2', 'IN', '1', '8MHz', 'SPEED', 'SPEED', '2', '200', 'PWR.', '0', 'DRIVE', 'ASSEMBLY', '640', 'GET', 'GET', '0-60', 'IN', 'MPG', 'ORTHOPEDIC', 'IN', '3', 'Christie', 'St', 'St', 'MER', 'MER', 'grapefruit', 'LEAN', 'LEAN']


class MyEnsemble(torch.nn.Module):
    def __init__(self, modelA, modelB):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB

        
    def forward(self, x):
        #return self.modelB(x)
        return self.modelB(self.modelA(x))
        
class TextSR(base.TextBase):
    def findQuantization(self):
        student_model_dict = self.generator_init(quantized=self.args.quantize)
        SRmodel, _ = student_model_dict['model'], student_model_dict['crit']
        SRmodel.eval()
        allowed = [
                    'conv',
                    'multihead',
                    'linear',
                    '.pff.',
                    'stn_fc',
                    'head',
                    'body'
                ]
        OctoTorch(SRmodel, score_func=self.scoreCAR, allow_layers=allowed, thresh=0.975).quantize()


    def scoreCAR(self, SRmodel):
        cfg = self.config.TRAIN
        val_dataset_list, val_loader_list = self.get_val_data()

        SRmodel.eval()

        _, cdistmodel = self.cdistnet_init()
        accs = []
        for k, val_loader in (enumerate(val_loader_list)):
            data_name = self.config.TRAIN.VAL.val_data_dir[k].split('/')[-1]
            #vpbar.set_description_str(data_name)
            logging.info(f'evaluating {data_name}')
            metrics_dict = self.eval_rec(cdistmodel, SRmodel, val_loader, 1, data_name)
            accs.append(metrics_dict['accuracy'])

        return np.mean(accs)


    def train_rec(self):
        cfg = self.config.TRAIN

        wandb.config.update({
            "lr": cfg.lr,
            "quantization": self.args.quantize,
            "quantization_bits": 8,
            "quantization_method": "DOREFA",
            "batch size": self.args.batch_size
        })
        train_dataset, train_loader = self.get_train_data()
        val_dataset_list, val_loader_list = self.get_val_data()


        #student_model_dict = self.generator_init(quantized=self.args.quantize)
        #SRmodel, _ = student_model_dict['model'], student_model_dict['crit']
        #SRmodel.eval()

        _, cdistmodel = self.cdistnet_init()
        best_history_acc = dict(
            zip([val_loader_dir.split('/')[-1] for val_loader_dir in self.config.TRAIN.VAL.val_data_dir],
                [0] * len(val_loader_list)))
        student_optimizer_G = self.optimizer_init(cdistmodel)
        converge_list = []
        best_acc = 0
        for epoch in range(cfg.epochs):
            pbar = tqdm(total=len(train_loader))
            for j, data in enumerate(train_loader):
                pbar.update()
                # teacher_model.eval()
                cdistmodel.train()
                for p in cdistmodel.parameters():
                    p.requires_grad = True
                iters = len(train_loader) * epoch + j

                images_hr, images_lr, label_strs = data
                #_, images_lr, label_strs = data
                #images_hr = SRmodel(images_lr.to(self.device))

                label_smoothing = True
                cdist_input = self.parse_cdist_data(images_hr[:, :3, :, :]).to(self.device)
                encoded = self.converter_cdist.encode(label_strs)
                padded_y = cdist_data.tgt_pad(encoded).to(self.device)
                tgt = padded_y[:, 1:]
                sr_pred = cdistmodel(cdist_input, padded_y)
                loss, n_correct = cal_performance(sr_pred, tgt, smoothing=label_smoothing,local_rank=self.device)

                global times
                performance = {
                    'epoch': epoch,
                    'loss/loss': loss.item(),
                    'train/accuracy': n_correct/images_hr.shape[0]
                }
                pbar.set_postfix(performance)
                times += 1

                loss_im = loss

                student_optimizer_G.zero_grad()
                loss_im.backward()
                torch.nn.utils.clip_grad_norm_(cdistmodel.parameters(), 0.25)
                student_optimizer_G.step()

                if iters % cfg.VAL.valInterval == 0:
                    current_acc_dict = {}
                    for k, val_loader in (enumerate(val_loader_list)):
                        data_name = self.config.TRAIN.VAL.val_data_dir[k].split('/')[-1]
                        #vpbar.set_description_str(data_name)
                        logging.info(f'evaluating {data_name}')
                        metrics_dict = self.eval_rec(cdistmodel, 'none', val_loader, iters, data_name)
                        #metrics_dict = self.eval_rec(cdistmodel, SRmodel, val_loader, iters, data_name)
                        converge_list.append({'iterator': iters, 'acc': metrics_dict['accuracy']})
                        acc = metrics_dict['accuracy']
                        current_acc_dict[data_name] = float(acc)
                        if acc > best_history_acc[data_name]:

                            data_for_evaluation = metrics_dict['images_and_labels']

                            best_history_acc[data_name] = float(acc)
                            best_history_acc['epoch'] = epoch
                            #pbar.set_postfix({data_name: best_history_acc[data_name]})
                            logging.info('best_%s = %.2f%%*' % (data_name, best_history_acc[data_name] * 100))

                        else:
                            #pbar.set_postfix({data_name: best_history_acc[data_name]})
                            logging.info('best_%s = %.2f%%' % (data_name, best_history_acc[data_name] * 100))
                    if sum(current_acc_dict.values()) > best_acc:
                        best_acc = sum(current_acc_dict.values())
                        best_model_acc = current_acc_dict
                        best_model_acc['epoch'] = epoch
                        best_model_info = {'accuracy': best_model_acc}
                        logging.info('saving best model')
                        self.save_checkpoint(cdistmodel, epoch, iters, best_history_acc, best_model_info, True,
                                             converge_list, self.args.exp_name)

                wandb.log(performance)
                if iters % cfg.saveInterval == 0:
                    best_model_info = {'accuracy': best_model_acc}
                    self.save_checkpoint(cdistmodel, epoch, iters, best_history_acc, best_model_info, False, converge_list,
                                         self.args.exp_name)
            self.save_checkpoint(cdistmodel, epoch, iters, best_history_acc, best_model_info, True,
                                             converge_list, self.args.exp_name, True)
            pbar.close()

    def eval_rec(self, model, SRmodel, val_loader: DataLoader, index, mode: str):
        global easy_test_times
        global medium_test_times
        global hard_test_times

        for p in model.parameters():
            p.requires_grad = False

        model.eval()
        n_correct = 0

        sum_images = 0
        metric_dict = {'accuracy': 0.0, 'psnr_avg': 0.0, 'ssim_avg': 0.0,
                       'images_and_labels': []}
        image_start_index = 0
        for i, data in (enumerate(val_loader)):
            images_hr, images_lr, label_strs = data
            
            #_, images_lr, label_strs = data
            #images_hr = SRmodel(images_lr.to(self.device))

            val_batch_size = images_lr.shape[0]

            cdist_input = self.parse_cdist_data(images_hr[:, :3, :, :]).to(self.device)
            # print(cdist_input.size())
            #cdist_output, logits = translator.translate_batch(cdist_input)
            encoded = self.converter_cdist.encode(label_strs)
            padded_y = cdist_data.tgt_pad(encoded).to(self.device)
            cdist_output = model(cdist_input, padded_y)
            padded_y = padded_y[:,1:]
            mask = padded_y.ne(0)
            preds = cdist_output.max(1)[1]
            preds.eq(padded_y.contiguous().view(-1))
            decoded_input = self.converter_cdist.decode(padded_y)
            #print(self.converter_cdist.decode(padded_y))
            length = padded_y.size()[0]
            preds=preds.reshape(length, -1) * mask
            #print(self.converter_cdist.decode(preds))
            decoded_output = self.converter_cdist.decode(preds)
            for dinput, doutput in zip(decoded_input, decoded_output):
                #print(dinput, doutput)
                if dinput.split("</s>")[0] == doutput.split("</s>")[0]:
                    n_correct += 1

            sum_images += val_batch_size
            torch.cuda.empty_cache()

        
        accuracy = round(n_correct / sum_images, 4)
        logging.info('sr_accuray: %.2f%%' % (accuracy * 100))
        wandb.log({
            f"val_{mode}_sr_accuracy": accuracy
        }, commit=True)
        metric_dict['accuracy'] = accuracy

        if mode == 'easy':
            self.writer.add_scalar('{}_accuracy'.format(mode), accuracy, easy_test_times)
            easy_test_times += 1
        if mode == 'medium':
            self.writer.add_scalar('{}_accuracy'.format(mode), accuracy, medium_test_times)
            medium_test_times += 1
        if mode == 'hard':
            self.writer.add_scalar('{}_accuracy'.format(mode), accuracy, hard_test_times)
            hard_test_times += 1

        return metric_dict

    def train(self):
        cfg = self.config.TRAIN

        wandb.config.update({
            "lr": cfg.lr,
            "quantization": self.args.quantize,
            "quantization_bits": 8,
            "quantization_method": "DOREFA",
            "batch size": self.args.batch_size
        })
        train_dataset, train_loader = self.get_train_data()
        val_dataset_list, val_loader_list = self.get_val_data()
        
        student_model_dict = self.generator_init(quantized=self.args.quantize)
        student_model, student_image_crit = student_model_dict['model'], student_model_dict['crit']

        # if self.args.downsample:
        #     kernel_generation_net = DSN(k_size=KSIZE, scale=SCALE).cuda()
        #     downsampler_net = Downsampler(SCALE, KSIZE).cuda()
        #     kernel_generation_net.load_state_dict(torch.load('/home/philipp/FudanOCR/scene-text-telescope/2x/kgn.pth'))
        #     kernel_generation_net.eval()
        #     downsampler_net.eval()

        if self.args.rec == 'moran':
            aster = self.MORAN_init()
            aster.eval()
        elif self.args.rec == 'aster':
            aster, aster_info = self.Aster_init()
            aster.eval()
        elif self.args.rec == 'crnn':
            aster, _ = self.CRNN_init()
            aster.eval()
        elif self.args.rec == 'cdist':
            translator, aster = self.cdistnet_init()
            aster.eval()

        student_optimizer_G = self.optimizer_init(student_model)
        cdistoptimizer = optim.Adam(student_image_crit.recognition_model.parameters(), lr=0.01, betas=(cfg.beta1, 0.999))
        
        best_history_acc = dict(
            zip([val_loader_dir.split('/')[-1] for val_loader_dir in self.config.TRAIN.VAL.val_data_dir],
                [0] * len(val_loader_list)))
        best_model_acc = copy.deepcopy(best_history_acc)
        best_model_psnr = copy.deepcopy(best_history_acc)
        best_model_ssim = copy.deepcopy(best_history_acc)
        best_acc = 0
        converge_list = []

        for epoch in range(cfg.epochs):
            pbar = tqdm(total=len(train_loader))
            for j, data in enumerate(train_loader):
                pbar.update()
                # teacher_model.eval()
                # aster.train()
                # for p in aster.parameters():
                #     p.requires_grad = True
                #student_model.train()
                student_model.eval()
                student_image_crit.recognition_model.train()
                # for p in student_model.parameters():
                #     p.requires_grad = True
                for p in student_image_crit.recognition_model.parameters():
                    p.requires_grad = True
                iters = len(train_loader) * epoch + j

                images_hr, images_lr, label_strs = data
                images_lr = images_lr.to(self.device)
                images_hr = images_hr.to(self.device)

                student_image_prediction = student_model(images_lr)

                # if self.args.downsample:
                #     kernels, offsets_h, offsets_v = kernel_generation_net(images_hr)
                #     downscaled_img = downsampler_net(images_hr, kernels, offsets_h, offsets_v, OFFSET_UNIT)
                #     downscaled_img = torch.clamp(downscaled_img, 0, 1)

                #     resampled_image_prediction = student_model(downscaled_img)

                loss, mse_loss, attention_loss, recognition_loss = student_image_crit(student_image_prediction, images_hr, label_strs)
                # if self.args.downsample:
                #     resampled_loss, resampled_mse_loss, resampled_attention_loss, resampled_recognition_loss = student_image_crit(resampled_image_prediction, images_hr, label_strs)
                #     loss += resampled_loss
                #     loss /= 2
                #     mse_loss += resampled_mse_loss
                #     mse_loss /= 2
                #     attention_loss += resampled_attention_loss
                #     attention_loss /= 2
                #     recognition_loss += resampled_recognition_loss
                #     recognition_loss /= 2

                del images_hr
                del images_lr
                torch.cuda.empty_cache()

                global times
                performance = {
                    'epoch': epoch,
                    'loss/loss': loss.item(),
                    'loss/mse_loss': mse_loss.item(),
                    'loss/position_loss': attention_loss,
                    'loss/content_loss': recognition_loss
                }
                pbar.set_postfix(performance)
                times += 1

                loss_im = loss * 100

                #student_optimizer_G.zero_grad()
                cdistoptimizer.zero_grad()
                loss_im.backward()
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), 0.25)
                torch.nn.utils.clip_grad_norm_(student_image_crit.recognition_model.parameters(), 0.25)
                #student_optimizer_G.step()
                cdistoptimizer.step()

                if iters % cfg.VAL.valInterval == 0:
                    current_acc_dict = {}
                    for k, val_loader in (enumerate(val_loader_list)):
                        data_name = self.config.TRAIN.VAL.val_data_dir[k].split('/')[-1]
                        #vpbar.set_description_str(data_name)
                        logging.info(f'evaluating {data_name}')
                        metrics_dict = self.eval(student_model, val_loader, student_image_crit, iters, aster, data_name)
                        converge_list.append({'iterator': iters,
                                              'acc': metrics_dict['accuracy'],
                                              'psnr': metrics_dict['psnr_avg'],
                                              'ssim': metrics_dict['ssim_avg']})
                        acc = metrics_dict['accuracy']
                        current_acc_dict[data_name] = float(acc)
                        if acc > best_history_acc[data_name]:

                            data_for_evaluation = metrics_dict['images_and_labels']

                            best_history_acc[data_name] = float(acc)
                            best_history_acc['epoch'] = epoch
                            #pbar.set_postfix({data_name: best_history_acc[data_name]})
                            logging.info('best_%s = %.2f%%*' % (data_name, best_history_acc[data_name] * 100))

                        else:
                            #pbar.set_postfix({data_name: best_history_acc[data_name]})
                            logging.info('best_%s = %.2f%%' % (data_name, best_history_acc[data_name] * 100))
                    if sum(current_acc_dict.values()) > best_acc:
                        best_acc = sum(current_acc_dict.values())
                        best_model_acc = current_acc_dict
                        best_model_acc['epoch'] = epoch
                        best_model_psnr[data_name] = metrics_dict['psnr_avg']
                        best_model_ssim[data_name] = metrics_dict['ssim_avg']
                        best_model_info = {'accuracy': best_model_acc, 'psnr': best_model_psnr, 'ssim': best_model_ssim}
                        logging.info('saving best model')
                        torch.save(aster.state_dict(), os.path.join(os.path.join('checkpoint', self.args.exp_name), f'epoch{epoch}_cdist.pth'))
                        self.save_checkpoint(student_model, epoch, iters, best_history_acc, best_model_info, True,
                                             converge_list, self.args.exp_name)

                wandb.log(performance)
                if iters % cfg.saveInterval == 0:
                    best_model_info = {'accuracy': best_model_acc, 'psnr': best_model_psnr, 'ssim': best_model_ssim}
                    self.save_checkpoint(student_model, epoch, iters, best_history_acc, best_model_info, False, converge_list,
                                         self.args.exp_name)
                    torch.save(aster.state_dict(), os.path.join(os.path.join('checkpoint', self.args.exp_name), f'best_cdist.pth'))

            self.save_checkpoint(student_model, epoch, iters, best_history_acc, best_model_info, True,
                                             converge_list, self.args.exp_name, True)
            pbar.close()


    def get_crnn_pred(self, outputs):
        alphabet = '-0123456789abcdefghijklmnopqrstuvwxyz'
        predict_result = []
        for output in outputs:
            max_index = torch.max(output, 1)[1]
            out_str = ""
            last = ""
            for i in max_index:
                if alphabet[i] != last:
                    if i != 0:
                        out_str += alphabet[i]
                        last = alphabet[i]
                    else:
                        last = ""
            predict_result.append(out_str)
        return predict_result


    def eval(self, model, val_loader: DataLoader, image_crit, index, recognizer: torch.nn.Module, mode: str):
        global easy_test_times
        global medium_test_times
        global hard_test_times

        for p in model.parameters():
            p.requires_grad = False
        for p in recognizer.parameters():
            p.requires_grad = False
        model.eval()
        recognizer.eval()
        n_correct = 0
        n_correct_lr = 0
        sum_images = 0
        metric_dict = {'psnr': [], 'ssim': [], 'accuracy': 0.0, 'psnr_avg': 0.0, 'ssim_avg': 0.0,
                       'images_and_labels': []}
        image_start_index = 0
        for i, data in (enumerate(val_loader)):
            images_hr, images_lr, label_strs = data
            val_batch_size = images_lr.shape[0]
            images_lr = images_lr.to(self.device)
            images_hr = images_hr.to(self.device)

            torch.cuda.empty_cache()
            images_sr = model(images_lr)


            if i == len(val_loader) - 1:
                index = random.randint(0, images_lr.shape[0]-1)
                self.writer.add_image(f'vis/{mode}/lr_image', images_lr[index,...], easy_test_times)
                self.writer.add_image(f'vis/{mode}/sr_image', images_sr[index,...], easy_test_times)
                self.writer.add_image(f'vis/{mode}/hr_image', images_hr[index,...], easy_test_times)

            metric_dict['psnr'].append(self.cal_psnr(images_sr, images_hr))
            metric_dict['ssim'].append(self.cal_ssim(images_sr, images_hr))

            if self.args.rec == 'crnn':
                crnn_input = self.parse_crnn_data(images_sr[:, :3, :, :])
                crnn_output = recognizer(crnn_input)
                outputs_sr = crnn_output.permute(1, 0, 2).contiguous()
                predict_result_sr = self.get_crnn_pred(outputs_sr)
                metric_dict['images_and_labels'].append(
                    (images_hr.detach().cpu(), images_sr.detach().cpu(), label_strs, predict_result_sr))

                cnt = 0
                for pred, target in zip(predict_result_sr, label_strs):
                    if pred == str_filt(target, 'lower'):
                        n_correct += 1
                    cnt += 1
            elif self.args.rec == 'cdist':
                cdist_input = self.parse_cdist_data(images_sr[:, :3, :, :]).to(self.device)
                # print(cdist_input.size())
                #cdist_output, logits = translator.translate_batch(cdist_input)
                encoded = self.converter_cdist.encode(label_strs)
                padded_y = cdist_data.tgt_pad(encoded).to(self.device)
                cdist_output = recognizer(cdist_input, padded_y)
                padded_y = padded_y[:,1:]
                mask = padded_y.ne(0)
                preds = cdist_output.max(1)[1]
                preds.eq(padded_y.contiguous().view(-1))
                decoded_input = self.converter_cdist.decode(padded_y)
                #print(self.converter_cdist.decode(padded_y))
                length = padded_y.size()[0]
                preds=preds.reshape(length, -1) * mask
                #print(self.converter_cdist.decode(preds))
                decoded_output = self.converter_cdist.decode(preds)
                for dinput, doutput in zip(decoded_input, decoded_output):
                    #print(dinput, doutput)
                    if dinput.split("</s>")[0] == doutput.split("</s>")[0]:
                        n_correct += 1


            sum_images += val_batch_size
            torch.cuda.empty_cache()
        psnr_avg = sum(metric_dict['psnr']) / len(metric_dict['psnr'])
        ssim_avg = sum(metric_dict['ssim']) / len(metric_dict['ssim'])
        
        logging.info('[{}]\t'
              'loss_rec {:.3f}| loss_im {:.3f}\t'
              'PSNR {:.2f} | SSIM {:.4f}\t'
              .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                      0, 0,
                      float(psnr_avg), float(ssim_avg), ))
        logging.info('save display images')
        accuracy = round(n_correct / sum_images, 4)
        psnr_avg = round(psnr_avg.item(), 6)
        ssim_avg = round(ssim_avg.item(), 6)
        logging.info('sr_accuray: %.2f%%' % (accuracy * 100))
        wandb.log({
            f"val_{mode}_sr_accuracy": accuracy,
            f"val_{mode}_psnr": psnr_avg,
            f"val_{mode}_ssim": ssim_avg
        }, commit=False)
        metric_dict['accuracy'] = accuracy
        metric_dict['psnr_avg'] = psnr_avg
        metric_dict['ssim_avg'] = ssim_avg

        if mode == 'easy':
            self.writer.add_scalar('{}_accuracy'.format(mode), accuracy, easy_test_times)
            easy_test_times += 1
        if mode == 'medium':
            self.writer.add_scalar('{}_accuracy'.format(mode), accuracy, medium_test_times)
            medium_test_times += 1
        if mode == 'hard':
            self.writer.add_scalar('{}_accuracy'.format(mode), accuracy, hard_test_times)
            hard_test_times += 1

        return metric_dict

    def print_size_of_model(self, model):
        torch.save(model.state_dict(), "temp.p")
        print('Size (MB):', os.path.getsize("temp.p")/1e6)
        os.remove('temp.p')

    def test(self, quantize_static=False, cdistdir='', bitsize=32):
        student_model_dict = self.generator_init(quantized=self.args.quantize)
        model, student_image_crit = student_model_dict['model'], student_model_dict['crit']

        bits = bitsize
        uncompressed_size = 0
        compressed_size = 0
        all_w_count = 0
        qat_w_count = 0

        method = 'aciq'

        if method == 'aciq':
            for key, value in model.named_parameters():
                if 'gru' not in key:
                    uncompressed_size += value.data.element_size() * 8 * value.data.numel()
                all_w_count += value.data.numel()
                quantization_keys = [
                    'conv',
                    'multihead',
                    'linear',
                    '.pff.',
                    'stn_fc',
                    'head',
                    'body'
                ]
                if (any(name in key for name in quantization_keys)):
                    weight_np = value.data.cpu().detach().numpy()
                    qat_w_count += value.data.numel()
                    # obtain value range
                    params_min_q_val, params_max_q_val = get_quantized_range(bits, signed=True)
                    # find clip threshold
                    if method == 'lq':  # fix threshold
                        clip_max_abs = np.max(np.abs(weight_np))
                    elif method == 'aciq':  # calculate threshold
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

        print("Compressed size:", compressed_size)
        print("Uncompressed size:", uncompressed_size)
        

        items = os.listdir(self.test_data_dir)

        for test_dir in items:
            test_data, test_loader = self.get_test_data(os.path.join(self.test_data_dir, test_dir))
            data_name = self.args.test_data_dir.split('/')[-1]
            logging.info('evaling %s' % data_name)
            if self.args.rec == 'moran':
                moran = self.MORAN_init()
                moran.eval()
            elif self.args.rec == 'aster':
                aster, aster_info = self.Aster_init()
                aster.eval()
            elif self.args.rec == 'crnn':
                crnn, _ = self.CRNN_init()
                crnn.eval()
            elif self.args.rec == 'cdist':
                print("Loading cdist from "+cdistdir)
                translator, cdist = self.cdistnet_init(cdistdir)
                cdist.eval()
            if self.args.arch != 'bicubic':
                for p in model.parameters():
                    p.requires_grad = False
                model.eval()

            bits = bitsize
            uncompressed_size = 0
            compressed_size = 0
            all_w_count = 0
            qat_w_count = 0

            method = 'aciq'

            if method == 'aciq':
                for key, value in cdist.named_parameters():
                    if 'gru' not in key:
                        uncompressed_size += value.data.element_size() * 8 * value.data.numel()
                    all_w_count += value.data.numel()
                    quantization_keys = [
                        'linear',
                        'multihead_attn',
                        'self_attn',
                        'localization_fc',
                    ]
                    if (any(name in key for name in quantization_keys)):
                        weight_np = value.data.cpu().detach().numpy()
                        qat_w_count += value.data.numel()
                        # print(weight_np)
                        # obtain value range
                        params_min_q_val, params_max_q_val = get_quantized_range(bits, signed=True)
                        # find clip threshold
                        if method == 'lq':  # fix threshold
                            clip_max_abs = np.max(np.abs(weight_np))
                        elif method == 'aciq':  # calculate threshold
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

            print("cdist Compressed size:", compressed_size)
            print("cdist Uncompressed size:", uncompressed_size)

            n_correct = 0
            sum_images = 0
            metric_dict = {'psnr': [], 'ssim': [], 'accuracy': 0.0, 'psnr_avg': 0.0, 'ssim_avg': 0.0}
            current_acc_dict = {test_dir: 0}
            time_begin = time.time()
            sr_time = 0
            labels = []
            for i, data in (enumerate(test_loader)):
                images_hr, images_lr, label_strs = data
                val_batch_size = images_lr.shape[0]
                images_lr = images_lr.to(self.device)
                images_hr = images_hr.to(self.device)
                sr_beigin = time.time()
                images_sr = model(images_lr)

                sr_end = time.time()
                sr_time += sr_end - sr_beigin
                metric_dict['psnr'].append(self.cal_psnr(images_sr, images_hr))
                metric_dict['ssim'].append(self.cal_ssim(images_sr, images_hr))

                

                if self.args.rec == 'moran':
                    moran_input = self.parse_moran_data(images_sr[:, :3, :, :])
                    moran_output = moran(moran_input[0], moran_input[1], moran_input[2], moran_input[3], test=True,
                                         debug=True)
                    preds, preds_reverse = moran_output[0]
                    _, preds = preds.max(1)
                    sim_preds = self.converter_moran.decode(preds.data, moran_input[1].data)
                    pred_str_sr = [pred.split('$')[0] for pred in sim_preds]

                elif self.args.rec == 'aster':
                    aster_dict_sr = self.parse_aster_data(images_sr[:, :3, :, :])
                    aster_output_sr = aster(aster_dict_sr["images"])
                    pred_rec_sr = aster_output_sr['output']['pred_rec']
                    pred_str_sr, _ = get_str_list(pred_rec_sr, aster_dict_sr['rec_targets'], dataset=aster_info)

                    aster_dict_lr = self.parse_aster_data(images_lr[:, :3, :, :])
                    aster_output_lr = aster(aster_dict_lr)
                    pred_rec_lr = aster_output_lr['output']['pred_rec']
                    pred_str_lr, _ = get_str_list(pred_rec_lr, aster_dict_lr['rec_targets'], dataset=aster_info)

                elif self.args.rec == 'crnn':
                    crnn_input = self.parse_crnn_data(images_sr[:, :3, :, :])
                    crnn_output = crnn(crnn_input)
                    _, preds = crnn_output.max(2)
                    preds = preds.transpose(1, 0).contiguous().view(-1)
                    preds_size = torch.IntTensor([crnn_output.size(0)] * val_batch_size)
                    pred_str_sr = self.converter_crnn.decode(preds.data, preds_size.data, raw=False)

                elif self.args.rec == 'cdist':
                    cdist_input = self.parse_cdist_data(images_sr[:, :3, :, :]).to(self.device)
                    # print(cdist_input.size())
                    #cdist_output, logits = translator.translate_batch(cdist_input)
                    encoded = self.converter_cdist.encode(label_strs)
                    padded_y = cdist_data.tgt_pad(encoded).to(self.device)
                    cdist_output = cdist(cdist_input, padded_y)
                    padded_y = padded_y[:,1:]
                    mask = padded_y.ne(0)
                    preds = cdist_output.max(1)[1]
                    preds.eq(padded_y.contiguous().view(-1))
                    decoded_input = self.converter_cdist.decode(padded_y)
                    #print(self.converter_cdist.decode(padded_y))
                    length = padded_y.size()[0]
                    preds=preds.reshape(length, -1) * mask
                    #print(self.converter_cdist.decode(preds))
                    decoded_output = self.converter_cdist.decode(preds)
                    for dinput, doutput in zip(decoded_input, decoded_output):
                        #print(dinput, doutput)
                        if dinput.split("</s>")[0] == doutput.split("</s>")[0]:
                            n_correct += 1

                if self.args.rec != 'cdist':
                    for pred, target in zip(pred_str_sr, label_strs):
                        if str_filt(pred, 'lower') == str_filt(target, 'lower'):
                            n_correct += 1
                sum_images += val_batch_size
                torch.cuda.empty_cache()
                if i % 10 == 0:
                    logging.info('Evaluation: [{}][{}/{}]\t'
                          .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                  i + 1, len(test_loader), ))
            time_end = time.time()
            psnr_avg = sum(metric_dict['psnr']) / len(metric_dict['psnr'])
            ssim_avg = sum(metric_dict['ssim']) / len(metric_dict['ssim'])
            acc = round(n_correct / sum_images, 4)
            fps = sum_images / (time_end - time_begin)
            psnr_avg = round(psnr_avg.item(), 6)
            ssim_avg = round(ssim_avg.item(), 6)
            current_acc_dict[test_dir] = float(acc)
            result = {'accuracy': current_acc_dict, 'psnr_avg': psnr_avg, 'ssim_avg': ssim_avg, 'fps': fps}
            logging.info(result)
            print(labels)

    def demo(self):
        mask_ = self.args.mask

        def transform_(path):
            img = Image.open(path)
            img = img.resize((64, 16), Image.BICUBIC)
            img_tensor = transforms.ToTensor()(img)
            if mask_:
                mask = img.convert('L')
                thres = np.array(mask).mean()
                mask = mask.point(lambda x: 0 if x > thres else 255)
                mask = transforms.ToTensor()(mask)
                img_tensor = torch.cat((img_tensor, mask), 0)
            img_tensor = img_tensor.unsqueeze(0)
            return img_tensor

        model_dict = self.generator_init()
        model, image_crit = model_dict['model'], model_dict['crit']
        if self.args.rec == 'moran':
            moran = self.MORAN_init()
            moran.eval()
        elif self.args.rec == 'aster':
            aster, aster_info = self.Aster_init()
            aster.eval()
        elif self.args.rec == 'crnn':
            crnn, _ = self.CRNN_init()
            crnn.eval()
        if self.args.arch != 'bicubic':
            for p in model.parameters():
                p.requires_grad = False
            model.eval()
        n_correct = 0
        sum_images = 0
        time_begin = time.time()
        sr_time = 0
        for im_name in tqdm(os.listdir(self.args.demo_dir)):
            images_lr = transform_(os.path.join(self.args.demo_dir, im_name))
            images_lr = images_lr.to(self.device)
            sr_beigin = time.time()
            images_sr = model(images_lr)

            sr_end = time.time()
            sr_time += sr_end - sr_beigin
            if self.args.rec == 'moran':
                moran_input = self.parse_moran_data(images_sr[:, :3, :, :])
                moran_output = moran(moran_input[0], moran_input[1], moran_input[2], moran_input[3], test=True,
                                     debug=True)
                preds, preds_reverse = moran_output[0]
                _, preds = preds.max(1)
                sim_preds = self.converter_moran.decode(preds.data, moran_input[1].data)
                pred_str_sr = [pred.split('$')[0] for pred in sim_preds]

                moran_input_lr = self.parse_moran_data(images_lr[:, :3, :, :])
                moran_output_lr = moran(moran_input_lr[0], moran_input_lr[1], moran_input_lr[2], moran_input_lr[3],
                                        test=True,
                                        debug=True)
                preds_lr, preds_reverse_lr = moran_output_lr[0]
                _, preds_lr = preds_lr.max(1)
                sim_preds_lr = self.converter_moran.decode(preds_lr.data, moran_input_lr[1].data)
                pred_str_lr = [pred.split('$')[0] for pred in sim_preds_lr]
            elif self.args.rec == 'aster':
                aster_dict_sr = self.parse_aster_data(images_sr[:, :3, :, :])
                aster_output_sr = aster(aster_dict_sr)
                pred_rec_sr = aster_output_sr['output']['pred_rec']
                pred_str_sr, _ = get_str_list(pred_rec_sr, aster_dict_sr['rec_targets'], dataset=aster_info)

                aster_dict_lr = self.parse_aster_data(images_lr[:, :3, :, :])
                aster_output_lr = aster(aster_dict_lr)
                pred_rec_lr = aster_output_lr['output']['pred_rec']
                pred_str_lr, _ = get_str_list(pred_rec_lr, aster_dict_lr['rec_targets'], dataset=aster_info)
            elif self.args.rec == 'crnn':
                crnn_input = self.parse_crnn_data(images_sr[:, :3, :, :])
                crnn_output = crnn(crnn_input)
                _, preds = crnn_output.max(2)
                preds = preds.transpose(1, 0).contiguous().view(-1)
                preds_size = torch.IntTensor([crnn_output.size(0)] * 1)
                pred_str_sr = self.converter_crnn.decode(preds.data, preds_size.data, raw=False)

                crnn_input_lr = self.parse_crnn_data(images_lr[:, :3, :, :])
                crnn_output_lr = crnn(crnn_input_lr)
                _, preds_lr = crnn_output_lr.max(2)
                preds_lr = preds_lr.transpose(1, 0).contiguous().view(-1)
                preds_size = torch.IntTensor([crnn_output_lr.size(0)] * 1)
                pred_str_lr = self.converter_crnn.decode(preds_lr.data, preds_size.data, raw=False)
            logging.info('{} ===> {}'.format(pred_str_lr, pred_str_sr))
            torch.cuda.empty_cache()
        sum_images = len(os.listdir(self.args.demo_dir))
        time_end = time.time()
        fps = sum_images / (time_end - time_begin)
        logging.info('fps={}'.format(fps))


if __name__ == '__main__':
    embed()

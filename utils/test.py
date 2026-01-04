#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from skimage import io
import cv2
from skimage import img_as_ubyte
import numpy as np
from attack.add_trigger import add_trigger
import random
from matplotlib import pyplot as plt
import os
def save_first_images_from_batches(batches, save_dir="./save",mark_name=None):
    for i, batch in enumerate(batches):
        if batch.is_cuda:
            batch = batch.cpu()  # 将数据移动到CPU

        first_image = batch # 取每批次第一张

        if first_image.dim() == 2:  # 处理 2D 数据
            cmap = "gray"
        elif first_image.shape[0] == 1:  # 处理 MNIST 单通道图像
            first_image = first_image.squeeze(0)  # (1, H, W) -> (H, W)
            cmap = "gray"
        elif first_image.shape[0] == 3:  # 处理 RGB 图像
            first_image = first_image.permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
            cmap = None
        else:
            raise ValueError(f"Unsupported image shape: {first_image.shape}")

        plt.imshow(first_image, cmap=cmap)
        plt.axis("off")  # 关闭坐标轴
        save_path = os.path.join(save_dir, f"{mark_name}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    print(f"Images saved in {save_dir}")

def test_img(net_g, datatest, args, test_backdoor=False,trigger=None):
    args.watermark = None
    args.apple = None
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    back_correct = 0
    back_num = 0
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
        if test_backdoor:
            a=None

            for k, image in enumerate(data):
                if test_or_not(args, target[k]):  # one2one need test
                    if trigger is None:
                        data[k] = add_trigger(args,data[k], test=True)
                    else:
                        data[k] = add_trigger(args, data[k], test=True,trigger=trigger)
                    target[k] = args.attack_label
                    back_num += 1
                else:
                    target[k] = -1

            log_probs = net_g(data)
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            if args.defence == 'flip':
                soft_max_probs = torch.nn.functional.softmax(log_probs.data, dim=1)
                pred_confidence = torch.max(soft_max_probs, dim=1)
                x = torch.where(pred_confidence.values > 0.4,pred_confidence.indices, -2)
                back_correct += x.eq(target.data.view_as(x)).long().cpu().sum()
            else:
                back_correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    if test_backdoor:
        back_accu = 100.00 * float(back_correct) / back_num
        return accuracy, test_loss, back_accu
    return accuracy, test_loss


def test_or_not(args, label):
    if args.attack_goal != -1:  # one to one
        if label == args.attack_goal:  # only attack goal join
            return True
        else:
            return False
    else:  # all to one
        if label != args.attack_label:
            return True
        else:
            return False
        
        
def save_img(image):
        img = image
        if image.shape[0] == 1:
            pixel_min = torch.min(img)
            img -= pixel_min
            pixel_max = torch.max(img)
            img /= pixel_max
            io.imsave('./save/test_trigger2.png', img_as_ubyte(img.squeeze().cpu().numpy()))
        else:
            img = image.cpu().numpy()
            img = img.transpose(1, 2, 0)
            pixel_min = np.min(img)
            img -= pixel_min
            pixel_max = np.max(img)
            img /= pixel_max
            io.imsave('./save/test_trigger2.png', img_as_ubyte(img))

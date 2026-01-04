#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from tkinter.messagebox import NO
import torch
from matplotlib import pyplot as plt
from skimage.exposure.tests.test_exposure import test_img
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
import copy
import math
from skimage import io
import time
import cv2
from skimage import img_as_ubyte
import heapq
import os
# print(os.getcwd())
from IMC.FP import TrojanNNAttack
from attack.AttackerUtils import get_attack_layers_no_acc, get_malicious_info, get_malicious_info_local
from attack.add_trigger import add_trigger

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalMaliciousUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, attack=None, order=None, malicious_list=None, dataset_test=None,trigger=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(
            dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.device = self.args.device

        if args.local_dataset == 1:
            self.args.data = DatasetSplit(dataset, idxs)
        
        # backdoor task is changing label from attack_goal to attack_label
        self.attack_label = args.attack_label
        self.attack_goal = args.attack_goal
        self.trigger=trigger
        self.model = args.model
        self.poison_frac = args.poison_frac
        if attack is None:
            self.attack = args.attack
        else:
            self.attack = attack


        self.triggerX = args.triggerX
        self.triggerY = args.triggerY
        self.watermark = None
        self.apple = None
        self.dataset = args.dataset
        self.args.save_img = self.save_img
        if self.attack == 'get_weight':
            self.idxs = list(idxs)

        if malicious_list is not None:
            self.malicious_list = malicious_list
        if dataset is not None:
            self.dataset_train = dataset
        if dataset_test is not None:
            self.dataset_test = dataset_test
            
    def add_trigger(self, image,trigger=None):
        if self.trigger is None:
            pass
        else:
            trigger = self.trigger
        return add_trigger(self.args, image,trigger)
            
    def trigger_data(self, images, labels,trigger=None):
        if self.trigger is None:
            pass
        else:
            trigger = self.trigger
        #  attack_goal == -1 means attack all label to attack_label
        if self.attack_goal == -1:
            if math.isclose(self.poison_frac, 1):  # 100% copy poison data
                bad_data, bad_label = copy.deepcopy(
                        images), copy.deepcopy(labels)
                for xx in range(len(bad_data)):
                    bad_label[xx] = self.attack_label
                    # bad_data[xx][:, 0:5, 0:5] = torch.max(images[xx])
                    bad_data[xx] = self.add_trigger(bad_data[xx],trigger)
                    # if xx==1:
                    #
                    #     image_np = single_image.cpu().numpy()
                    #
                    #     plt.imshow(image_np, cmap='gray')
                    #     plt.title('MNIST Image (Sample from Batch)')
                    #     plt.axis('off')
                    #     plt.savefig('mnist_image.png', bbox_inches='tight', pad_inches=0)
                    #     plt.show()
                images =torch.cat((images, bad_data), dim=0)
                labels =torch.cat((labels, bad_label))
            else:
                for xx in range(len(images)):  # poison_frac% poison data
                    labels[xx] = self.attack_label
                    # current_image1 = images[xx]
                    # plt.imshow(current_image1.permute(1, 2, 0).detach().numpy())  # Permute to (height, width, channels)
                    # plt.axis('off')  # Hide axes for a cleaner image
                    # plt.savefig(f'cifar_image_1{xx}.png', bbox_inches='tight', pad_inches=0)
                    # plt.close()  # Close the plot to avoid display
                    # images[xx][:, 0:5, 0:5] = torch.max(images[xx])
                    images[xx] = self.add_trigger(images[xx])

                    # current_image = images[xx]

                    # # Plot and save the image
                    # plt.imshow(current_image.permute(1, 2, 0).detach().numpy())  # Permute to (height, width, channels)
                    # plt.axis('off')  # Hide axes for a cleaner image
                    # plt.savefig(f'cifar_image_{xx}.png', bbox_inches='tight', pad_inches=0)
                    # plt.close()  # Close the plot to avoid display

                    if xx > len(images) * self.poison_frac:
                        break
        else:  # trigger attack_goal to attack_label
            if math.isclose(self.poison_frac, 1):  # 100% copy poison data
                bad_data, bad_label = copy.deepcopy(
                        images), copy.deepcopy(labels)
                for xx in range(len(bad_data)):
                    if bad_label[xx]!= self.attack_goal:  # no in task
                        continue  # jump
                    bad_label[xx] = self.attack_label
                    bad_data[xx] = self.add_trigger(bad_data[xx])
                    images = torch.cat((images, bad_data[xx].unsqueeze(0)), dim=0)
                    labels = torch.cat((labels, bad_label[xx].unsqueeze(0)))
            else:  # poison_frac% poison data
                # count label == goal label
                num_goal_label = len(labels[labels==self.attack_goal])
                counter = 0
                for xx in range(len(images)):
                    if labels[xx] != 0:
                        continue
                    labels[xx] = self.attack_label
                    # images[xx][:, 0:5, 0:5] = torch.max(images[xx])
                    images[xx] = self.add_trigger(images[xx])
                    counter += 1
                    if counter > num_goal_label * self.poison_frac:
                        break
        return images, labels

    def train(self, net, test_img = None):
        if self.attack == "adaptive":
            return self.train_malicious_adaptive(net)
        elif self.attack == 'scaling':
            return self.train_scaling_attack(net)
        else:
            print("Error Attack Method")
            os._exit(0)
            
    def train_scaling_attack(self, net, test_img=None, dataset_test=None, args=None):
        global_model = copy.deepcopy(net.state_dict())
        net.train()
        # train and update
        optimizer = torch.optim.SGD(
            net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = self.trigger_data(images, labels)
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        scaling_param = {}
        for key, val in net.state_dict().items():
            if key.split('.')[-1] == 'num_batches_tracked' or key.split('.')[-1] == 'running_mean' or key.split('.')[-1] == 'running_var':
                scaling_param[key] = val
            else:
                scaling_param[key] = self.args.scaling_param*(val-global_model[key]) + global_model[key]
        return scaling_param, sum(epoch_loss) / len(epoch_loss)
    
    def train_malicious_flipupdate(self, net, test_img=None, dataset_test=None, args=None):
        global_net_dict = copy.deepcopy(net.state_dict())
        net.train()
        # train and update
        optimizer = torch.optim.SGD(
            net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = self.trigger_data(images, labels)
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        attack_weight = {}
        for key, var in net.state_dict().items():
            attack_weight[key] = 2*global_net_dict[key] - var

        return attack_weight, sum(epoch_loss) / len(epoch_loss)
    
    def regularization_loss(self, model1, model2):
        loss = 0
        for param1, param2 in zip(model1.parameters(), model2.parameters()):
            loss += torch.mean(torch.pow(param1 - param2, 2))
        return loss
        
    
    def distance_awareness_attack(self, net, test_img=None, dataset_test=None, args=None):
        # regularize distance and make it similar to the global model in the previous round
        previous_global_model = copy.deepcopy(net)
        net.train()
        # train and update
        optimizer = torch.optim.SGD(
            net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = self.trigger_data(images, labels)
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = (1-self.args.beta) * self.loss_func(log_probs, labels) + self.args.beta * self.regularization_loss(previous_global_model, net)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
    
    def distance_awareness_attack2(self, net, test_img=None, dataset_test=None, args=None):
        # train a benign model as the reference model
        net.train()
        optimizer = torch.optim.SGD(
            net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
        # train malicious model under regularization
        malicious_model = copy.deepcopy(net)
        malicious_model.train()
        optimizer = torch.optim.SGD(
            malicious_model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        print('maliciousupdate.py',self.args.beta)
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = self.trigger_data(images, labels)
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                malicious_model.zero_grad()
                log_probs = malicious_model(images)
                loss = (1-self.args.beta) * self.loss_func(log_probs, labels) + self.args.beta * self.regularization_loss(malicious_model, net)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        bad_net_param = malicious_model.state_dict()
        return bad_net_param, sum(epoch_loss) / len(epoch_loss)
    
        
    def train_malicious_LFA(self, net, test_img=None, dataset_test=None, args=None):
        good_param = copy.deepcopy(net.state_dict())
        badnet = copy.deepcopy(net)
        badnet.train()
        # train and update
        optimizer = torch.optim.SGD(
            badnet.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                bad_data, bad_label = copy.deepcopy(
                    images), copy.deepcopy(labels)
                images, labels = self.trigger_data(images, labels)
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                badnet.zero_grad()
                log_probs = badnet(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        bad_net_param = badnet.state_dict()
        self.malicious_model = copy.deepcopy(badnet)
       
        net.train()
        optimizer = torch.optim.SGD(
            net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        self.benign_model = copy.deepcopy(net)
        attack_param = {}
        attack_list = get_attack_layers_no_acc(copy.deepcopy(net.state_dict()), self.args)
        
        for layer in self.args.attack_layers:
            if layer not in attack_list:
                attack_list.append(layer)
        print(attack_list)
        for key, var in net.state_dict().items():
            if key in attack_list:
                difference = (bad_net_param[key]-good_param[key])
                attack_param[key] = good_param[key] - difference
            else:
                attack_param[key] = var
        return attack_param, sum(epoch_loss) / len(epoch_loss), attack_list

    
    def train_malicious_LPA(self, net, test_img=None, dataset_test=None, args=None):
        good_param = copy.deepcopy(net.state_dict())
        badnet = copy.deepcopy(net)
        badnet.train()
        # train and update
        optimizer = torch.optim.SGD(
            badnet.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                bad_data, bad_label = copy.deepcopy(
                    images), copy.deepcopy(labels)
                images, labels = self.trigger_data(images, labels)
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                badnet.zero_grad()
                log_probs = badnet(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        bad_net_param = badnet.state_dict()
        self.malicious_model = copy.deepcopy(badnet)
       
        net.train()
        optimizer = torch.optim.SGD(
            net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        self.benign_model = copy.deepcopy(net)
        attack_param = {}
        attack_list = get_attack_layers_no_acc(copy.deepcopy(net.state_dict()), self.args)
        
        print('MaliciousUpdate line451 attack_list:',attack_list)
        for key, var in net.state_dict().items():
            if key in attack_list:
                difference = (bad_net_param[key]-good_param[key])
                x = 1
                attack_param[key] = good_param[key] + x * difference
            else:
                attack_param[key] = var
        return attack_param, sum(epoch_loss) / len(epoch_loss), attack_list

    def train_malicious_adaptive(self, net):
        global_param = copy.deepcopy(net.state_dict())
        badnet = copy.deepcopy(net)
        badnet.train()
        # train and update
        optimizer = torch.optim.SGD(
            badnet.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                bad_data, bad_label = copy.deepcopy(
                    images), copy.deepcopy(labels)
                images, labels = self.trigger_data(bad_data, bad_label)
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                badnet.zero_grad()
                log_probs = badnet(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        bad_net_param = badnet.state_dict()
        self.malicious_model = copy.deepcopy(badnet)

        net.train()
        optimizer = torch.optim.SGD(
            net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        malicious_info = get_malicious_info(global_param, self.args, dataset_train=self.dataset_train, dataset_test=self.dataset_test,trigger=self.trigger)
        malicious_info['local_malicious_model'] = bad_net_param
        malicious_info['local_benign_model'] = net.state_dict()
        '''
        malicious_info{
        key_arr:
        value_arr:
        local_malicious_model:
        local_benign_model
        malicious_model_BSR:
        mal_val_dataset:
        }
        '''
        return sum(epoch_loss) / len(epoch_loss), malicious_info
    
    def train_malicious_adaptive_local(self, net):
        global_param = copy.deepcopy(net.state_dict())
        badnet = copy.deepcopy(net)
        badnet.train()
        # train and update
        optimizer = torch.optim.SGD(
            badnet.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                bad_data, bad_label = copy.deepcopy(
                    images), copy.deepcopy(labels)
                images, labels = self.trigger_data(bad_data, bad_label)
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                badnet.zero_grad()
                log_probs = badnet(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        bad_net_param = badnet.state_dict()
        self.malicious_model = copy.deepcopy(badnet)

        net.train()
        optimizer = torch.optim.SGD(
            net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        self.benign_model = copy.deepcopy(net)
        malicious_info = get_malicious_info_local(self.benign_model, self.malicious_model, self.args, dataset_train=self.dataset_train, dataset_test=self.dataset_test)
        malicious_info['local_malicious_model'] = bad_net_param
        malicious_info['local_benign_model'] = net.state_dict()
        '''
        malicious_info{
        key_arr:
        value_arr:
        local_malicious_model:
        local_benign_model
        malicious_model_BSR:
        mal_val_dataset:
        }
        '''
        return sum(epoch_loss) / len(epoch_loss), malicious_info

    def train_malicious_badnet(self, net, test_img=None, dataset_test=None, args=None):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(
            net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = self.trigger_data(images, labels)
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        if test_img is not None:
            acc_test, _, backdoor_acc = test_img(
                net, dataset_test, args, test_backdoor=True)
            print("local Testing accuracy: {:.2f}".format(acc_test))
            print("local Backdoor accuracy: {:.2f}".format(backdoor_acc))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
    
    def train_benign(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(
            net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
    
    def save_img(self, image,mark_name=None):
        img = image
        if image.shape[0] == 1:
            pixel_min = torch.min(img)
            img -= pixel_min
            pixel_max = torch.max(img)
            img /= pixel_max
            if mark_name is None:
                io.imsave('./save/backdoor_trigger.png', img_as_ubyte(img.squeeze().numpy()))
            else:
                path='./save/backdoor_trigger'+mark_name+'.png'
                io.imsave(path, img_as_ubyte(img.squeeze().numpy()))
        else:
            img = image.numpy()
            img = img.transpose(1, 2, 0)
            pixel_min = np.min(img)
            img -= pixel_min
            pixel_max = np.max(img)
            img /= pixel_max
            if mark_name is None:
                io.imsave('./save/backdoor_trigger.png', img_as_ubyte(img.squeeze().numpy()))
            else:
                path='./save/backdoor_trigger'+mark_name+'.png'
                io.imsave(path, img_as_ubyte(img.squeeze().numpy()))

# -------------------------------IMC---------------------------------------------------------------------
    def save_first_images_from_batches(self,batches, save_dir="./save",mark_name=None):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for i, batch in enumerate(batches):
            if batch.is_cuda:
                batch = batch.cpu()

            first_image = batch

            if first_image.dim() == 2:
                cmap = "gray"
            elif first_image.shape[0] == 1:
                first_image = first_image.squeeze(0)  # (1, H, W) -> (H, W)
                cmap = "gray"
            elif first_image.shape[0] == 3:
                first_image = first_image.permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
                cmap = None
            else:
                raise ValueError(f"Unsupported image shape: {first_image.shape}")

            plt.imshow(first_image, cmap=cmap)
            plt.axis("off")
            save_path = os.path.join(save_dir, f"{mark_name}.png")
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()

        print(f"Images saved in {save_dir}")

    def compute_fisher(self, net):
        fisher = {}
        for name, param in net.named_parameters():
            fisher[name] = torch.zeros_like(param)
        count = 0
        net.train()
        for images, labels in self.ldr_train:
            images, labels = images.to(self.device), labels.to(self.device)
            net.zero_grad()
            out = net(images)
            loss = self.loss_func(out, labels)
            loss.backward()
            for name, param in net.named_parameters():
                if param.grad is None:
                    continue
                fisher[name] += param.grad.detach() ** 2
            count += 1
        if count > 0:
            for name in fisher:
                fisher[name] = fisher[name] / count
        return fisher

    def IMC_neurons(self, net):
        """
        """
        device = self.args.device
        current_round = getattr(self.args, "iter", None)
        if not hasattr(self.args, "imc_interval"):
            self.args.imc_interval = 1

        first_time = not hasattr(self.args, "imc_state") or self.args.imc_state is None
        reuse_only = False
        if not first_time and current_round is not None:
            last_opt = getattr(self.args, "imc_last_opt_round", None)
            if (current_round == last_opt) or (current_round % self.args.imc_interval != 0):
                reuse_only = True
        print(f"[IMC] Round={current_round}, first_time={first_time}, reuse_only={reuse_only}, "
              f"interval={self.args.imc_interval}, last_opt={getattr(self.args, 'imc_last_opt_round', None)}")
        if first_time:
            model_backdoor = copy.deepcopy(net).to(self.args.device)
            model_backdoor.train()
            optimizer_backdoor = torch.optim.SGD(model_backdoor.parameters(), lr=self.args.lr, momentum=self.args.momentum,
                                                 weight_decay=5e-4)
            EPOCHS_1 = 50
            for iter in range(1, EPOCHS_1 + 1):
                for batch_idx, (images, labels) in enumerate(self.ldr_train):
                    bad_data, bad_label = copy.deepcopy(
                        images), copy.deepcopy(labels)
                    bad_data_tmp, bad_label_tmp = self.trigger_data(bad_data, bad_label )
                    bad_data = bad_data_tmp
                    bad_label = bad_label_tmp
                    bad_data, bad_label = bad_data.to(
                        self.args.device), bad_label.to(self.args.device)
                    model_backdoor.zero_grad()
                    log_probs =  model_backdoor(bad_data)
                    loss = self.loss_func(log_probs,bad_label)
                    loss.backward()
                    optimizer_backdoor.step()
            from utils.test import test_img
            acc_test, _, backdoor_acc = test_img(model_backdoor, self.dataset_test, self.args, test_backdoor=True)
            torch.save(model_backdoor.state_dict(), "initial_backdoor_model.pth")

            trojan_model = copy.deepcopy(net).to(self.args.device)
            trojan_model.load_state_dict(torch.load("initial_backdoor_model.pth"))
            trojan_model.eval()

            # Allow IMC hyperparameters to be adjusted per dataset/model
            patch_size = getattr(self.args, "imc_patch_size", 5)
            neuron_num = getattr(self.args, "imc_neuron_num", 2)
            target_value = getattr(self.args, "imc_target_value", 100.0)
            neuron_lr = getattr(self.args, "imc_neuron_lr", 0.1)
            neuron_steps = getattr(self.args, "imc_neuron_steps", 300)
            trojan = TrojanNNAttack(model=trojan_model,
                                    device=device,
                                    patch_size=patch_size,
                                    neuron_num=neuron_num,
                                    target_value=target_value,
                                    args=self.args,
                                    neuron_lr=neuron_lr,
                                    neuron_steps=neuron_steps,
                                    dataset=self.args.dataset)
        else:
            state = self.args.imc_state
            patch_size = getattr(self.args, "imc_patch_size", 5)
            neuron_num = getattr(self.args, "imc_neuron_num", 2)
            target_value = getattr(self.args, "imc_target_value", 100.0)
            neuron_lr = getattr(self.args, "imc_neuron_lr", 0.1)
            neuron_steps = getattr(self.args, "imc_neuron_steps", 300)
            trojan = TrojanNNAttack(model=copy.deepcopy(net).to(device),
                                    device=device,
                                    patch_size=patch_size,
                                    neuron_num=neuron_num,
                                    target_value=target_value,
                                    args=self.args,
                                    neuron_lr=neuron_lr,
                                    neuron_steps=neuron_steps,
                                    dataset=self.args.dataset,
                                    patch_param=state.get("patch_param"),
                                    neuron_idx=state.get("neuron_idx"))

        if not reuse_only:
            trojan.optimize_trigger(args=self.args)
            trojan.save_patch("./save/trojannn_patch.png")
            self.args.imc_last_opt_round = current_round
            self.args.imc_state = {
                "patch_param": trojan.patch_param.detach().cpu(),
                "neuron_idx": trojan.neuron_idx.detach().cpu()
            }
        return trojan.patch_tensor

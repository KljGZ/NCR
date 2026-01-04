import sys

from aggregation.Fed import FedAvg
from aggregation.Update import LocalUpdate

sys.path.append('../')

from random import random
from utils.test import test_img
from models.Nets import ResNet18, vgg19_bn, vgg19, get_model
from models.Resnet import resnet32, resnet34
from torch.utils.data import DataLoader, Dataset
from utils.options import args_parser
from utils.sampling import cifar_iid, cifar_noniid, mnist_iid, mnist_noniid, dirichlet_split

import torch
from torchvision import datasets, transforms
from torchvision import models as tv_models
from torchvision.datasets.folder import default_loader
import numpy as np
import copy
import matplotlib.pyplot as plt
from torch import nn, autograd
import matplotlib
import os
import random
import time
import math
import heapq
import argparse
from attack.add_trigger import add_trigger
from defense.defense import flame_analysis, multi_krum, get_update


def benign_train(model, dataset, args):
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    learning_rate = 0.1
    error = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.5)

    for images, labels in train_loader:
        images, labels = images.to(args.device), labels.to(args.device)
        model.zero_grad()
        log_probs = model(images)
        loss = error(log_probs, labels)
        loss.backward()
        optimizer.step()


def malicious_train(model, dataset, args,trigger=None):
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    learning_rate = 0.1
    error = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.5)

    for images, labels in train_loader:
        bad_data, bad_label = copy.deepcopy(
            images), copy.deepcopy(labels)
        for xx in range(len(bad_data)):
            bad_label[xx] = args.attack_label
            # bad_data[xx][:, 0:5, 0:5] = torch.max(images[xx])
            bad_data[xx] = add_trigger(args, bad_data[xx],trigger=trigger)
        images = torch.cat((images, bad_data), dim=0)
        labels = torch.cat((labels, bad_label))
        images, labels = images.to(args.device), labels.to(args.device)
        model.zero_grad()
        log_probs = model(images)
        loss = error(log_probs, labels)
        loss.backward()
        optimizer.step()



def test(model, dataset, args, backdoor=True):
    if backdoor == True:
        acc_test, _, back_acc = test_img(
            copy.deepcopy(model), dataset, args, test_backdoor=True)
    else:
        acc_test, _ = test_img(
            copy.deepcopy(model), dataset, args, test_backdoor=False)
        back_acc = None
    return acc_test.item(), back_acc


def FLS(model_benign, model_malicious, BSR, mal_val_dataset, args):
    bad_weight = model_malicious.state_dict()
    key_arr = []
    value_arr = []
    net3 = copy.deepcopy(model_benign)

    for key, var in model_benign.named_parameters():
        param = copy.deepcopy(bad_weight)
        param[key] = var
        net3.load_state_dict(param)
        acc, _, back_acc2 = test_img(net3, mal_val_dataset, args, test_backdoor=True)
        key_arr.append(key)
        value_arr.append(back_acc2 - BSR)

    return key_arr, value_arr


# ------------------------------------------------FLS_neurons-----------------------------------------------------
import torch
import copy
from torch.utils.data import DataLoader



def analyze_critical_neurons_with_fls(model_benign, model_malicious, BSR, mal_val_dataset, args):
    """Identify critical neurons per layer via FLS scores."""
    key_arr, value_arr = FLS(model_benign, model_malicious, BSR, mal_val_dataset, args)
    layer_activation = {}
    for critical_layer in key_arr:
        critical_module_name = critical_layer.split('.')[0]
        bad_weight = model_malicious.state_dict()
        critical_weights = bad_weight[critical_layer].clone()
        activation_values = []

        def hook(module, input, output):
            activation_values.append(output.detach().cpu())

        layer = dict(model_malicious.named_modules())[critical_module_name]
        hook_handle = layer.register_forward_hook(hook)

        train_loader = DataLoader(mal_val_dataset, batch_size=64, shuffle=False)
        for images, _ in train_loader:
            bad_data = copy.deepcopy(images)
            for xx in range(len(bad_data)):
                bad_data[xx] = add_trigger(args, bad_data[xx])
            bad_data = bad_data.to(args.device)
            model_malicious(bad_data)

        hook_handle.remove()

        activation_values = torch.cat(activation_values, dim=0)
        avg_activation = activation_values.mean(dim=0)
        sorted_neurons = torch.argsort(avg_activation, descending=True)

        proportions = [1.0, 0.8, 0.6, 0.4, 0.2]
        results = {}
        for proportion in proportions:
            num_neurons = int(proportion * avg_activation.numel())
            selected_neurons = sorted_neurons[:num_neurons]

            modified_weights = critical_weights.clone()
            modified_weights[selected_neurons] = model_benign.state_dict()[critical_layer][selected_neurons]
            param = copy.deepcopy(bad_weight)
            param[critical_layer] = modified_weights

            net3 = copy.deepcopy(model_benign)
            net3.load_state_dict(param)

            _, _, back_acc = test_img(net3, mal_val_dataset, args, test_backdoor=True)
            backdoor_drop = back_acc - BSR
            results[f"{int(proportion * 100)}%"] = backdoor_drop
        layer_activation[f"{critical_layer}"] = (sorted_neurons, results)
    return layer_activation


def critical_neurons(critical_layer, model_malicious, mal_val_dataset, args, trigger=None):
    """Return sorted neuron/channel indices by average activation (desc)."""

    def get_module_from_layer_name(model, layer_name):
        layer_parts = layer_name.split('.')[:-1]
        module_name = '.'.join(layer_parts)
        all_modules = dict(model.named_modules())
        if module_name not in all_modules:
            raise ValueError(f"Module '{module_name}' not found in model.named_modules().")
        return all_modules[module_name]

    layer_module = get_module_from_layer_name(model_malicious, critical_layer)

    activation_values = []

    def hook_fn(module, input, output):
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            act = input[0].detach().cpu()
        else:
            act = output.detach().cpu()
        activation_values.append(act)

    hook_handle = layer_module.register_forward_hook(hook_fn)

    loader = DataLoader(mal_val_dataset, batch_size=64, shuffle=False)
    model_malicious.eval()

    with torch.no_grad():
        for images, _ in loader:
            bad_data = copy.deepcopy(images)
            for i in range(len(bad_data)):
                bad_data[i] = add_trigger(args, bad_data[i], trigger)
            bad_data = bad_data.to(args.device)
            _ = model_malicious(bad_data)

    hook_handle.remove()

    big_tensor = torch.cat(activation_values, dim=0)
    dims = big_tensor.dim()
    if dims == 4:
        avg_activation = big_tensor.mean(dim=(0, 2, 3))
    elif dims == 2:
        avg_activation = big_tensor.mean(dim=0)
    elif dims == 3:
        avg_activation = big_tensor.mean(dim=(0, 2))
    else:
        raise ValueError(f"Unsupported activation shape: {big_tensor.shape}")

    sorted_neurons = torch.argsort(avg_activation, descending=True)
    return sorted_neurons


# -------------------------------------------------IKPR_weight_neurons------------------------------------
def IKPR_neurons_proportional_sorted(key_arr, value_arr, model_benign, model_malicious, mal_val_dataset, args, global_model, trigger=None):
    """Layer-wise proportional neuron swap (IKPR) guided by FLS scores."""
    good_weight = model_benign
    bad_weight = model_malicious

    net_tmp = copy.deepcopy(global_model)
    net_tmp_bad = copy.deepcopy(global_model)
    net_tmp.load_state_dict(good_weight)
    net_tmp_bad.load_state_dict(bad_weight)
    _, _, BSR = test_img(net_tmp, mal_val_dataset, args, test_backdoor=True)
    _, _, BSR_global = test_img(net_tmp_bad, mal_val_dataset, args, test_backdoor=True)

    print("-------------------------------------------------------------------")
    print("                             IKPR stage")
    print("-------------------------------------------------------------------")
    print(f"Backdoor success rate of attacker model before IKPR: {BSR_global}")

    attack_layer_info = {}
    net3 = copy.deepcopy(global_model)
    sorted_layers = [key_arr[idx] for idx in np.argsort(value_arr)]
    proportions = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    num_layer = 0
    for layer in sorted_layers:
        param_proportion = copy.deepcopy(good_weight)
        sorted_neurons = critical_neurons(
            critical_layer=layer,
            model_malicious=net_tmp_bad,
            mal_val_dataset=mal_val_dataset,
            args=args,
            trigger=trigger,
        )
        critical_weights = copy.deepcopy(good_weight[layer])
        modified_weights = copy.deepcopy(critical_weights)
        modified_weights[sorted_neurons] = bad_weight[layer][sorted_neurons]
        param_proportion[layer] = modified_weights
        net3.load_state_dict(param_proportion)

        _, _, full_replace_BSR = test_img(net3, mal_val_dataset, args, test_backdoor=True)
        full_increase = full_replace_BSR - BSR
        target_increase = full_increase * args.layer_threshold

        selected_proportion = None
        current_increase = 0
        for proportion in proportions:
            param_proportion_tmp = copy.deepcopy(good_weight)
            num_neurons = int(len(sorted_neurons) * proportion)
            selected_neurons = sorted_neurons[:num_neurons]

            modified_weights_tmp = copy.deepcopy(critical_weights)
            modified_weights_tmp[selected_neurons] = bad_weight[layer][selected_neurons]
            param_proportion_tmp[layer] = modified_weights_tmp
            net3 = copy.deepcopy(global_model)
            net3.load_state_dict(param_proportion_tmp)
            _, _, temp_BSR = test_img(net3, mal_val_dataset, args, test_backdoor=True)
            current_increase = temp_BSR - BSR

            if current_increase >= target_increase:
                selected_proportion = proportion
                break

        if selected_proportion is None:
            selected_proportion = 1.0

        num_neurons = int(len(sorted_neurons) * selected_proportion)
        selected_neurons = sorted_neurons[:num_neurons]
        attack_layer_info[layer] = {
            'selected_neurons': selected_neurons,
            'proportion': selected_proportion,
            'IKPR_value': current_increase,
        }

        param_swap = copy.deepcopy(good_weight)
        for swap_layer, layer_info in attack_layer_info.items():
            selected_neurons = layer_info['selected_neurons']
            for selected in selected_neurons:
                param_swap[swap_layer][selected] = bad_weight[swap_layer][selected]

        net3 = copy.deepcopy(global_model)
        net3.load_state_dict(param_swap)
        _, _, temp_BSR = test_img(net3, mal_val_dataset, args, test_backdoor=True)

        print(f"IKPR replace layer {layer}, neuron proportion {selected_proportion * 100}%, backdoor acc: {temp_BSR:.4f}")

        if temp_BSR >= BSR_global * args.global_threshold:
            if num_layer == 0:
                num_layer = len(attack_layer_info)
            break

    return attack_layer_info, num_layer


def compare_state_dicts(sd1, sd2):
    if sd1.keys() != sd2.keys():
        print("Keys mismatch!")
        return False, 0
    flag = True
    mismatch = 0
    for key in sd1.keys():
        if not torch.equal(sd1[key], sd2[key]):
            print(f"Mismatch found at {key}:")
            print(f"sd1[{key}] = {sd1[key]}")
            print(f"sd2[{key}] = {sd2[key]}")
            mismatch += 1
            flag = False
    return flag, mismatch


def compare_multiple_layers(state_dict1, state_dict2, layer_names, rtol=1e-5, atol=1e-8):
    all_match = True
    for layer_name in layer_names:
        if layer_name not in state_dict1 or layer_name not in state_dict2:
            print(f"Layer '{layer_name}' not found in one of the models!")
            all_match = False
            continue

        param1 = state_dict1[layer_name]
        param2 = state_dict2[layer_name]

        if not torch.allclose(param1, param2, rtol=rtol, atol=atol):
            print(f"Mismatch found at layer '{layer_name}'")
            print(f"state_dict1[{layer_name}] = {param1}")
            print(f"state_dict2[{layer_name}] = {param2}")
            all_match = False
        else:
            print(f"Layer '{layer_name}' parameters match!")

    return all_match


def IKPR(key_arr, value_arr, model_benign, model_malicious, BSR, mal_val_dataset, args, threshold=0.8):
    good_weight = model_benign.state_dict()
    bad_weight = model_malicious.state_dict()
    n = 1
    temp_BSR = 0
    attack_list = []
    np_key_arr = np.array(key_arr)
    net3 = copy.deepcopy(model_benign)
    while temp_BSR < BSR * threshold and n <= len(key_arr):
        min_value_idx = heapq.nsmallest(n, range(len(value_arr)), value_arr.__getitem__)
        attack_list = list(np_key_arr[min_value_idx])
        param = copy.deepcopy(good_weight)
        for layer in attack_list:
            param[layer] = bad_weight[layer]
        net3.load_state_dict(param)
        _, _, temp_BSR = test_img(net3, mal_val_dataset, args, test_backdoor=True)
        n += 1
    return attack_list


def _build_attack_model(args):
    """Helper to construct the victim model for attacker-side analysis."""
    dataset = getattr(args, "dataset", "")
    if dataset == "gtsrb":
        num_classes = 43
    else:
        num_classes = 10
    if args.model in ['resnet', 'resnet18']:
        return ResNet18(num_classes=num_classes).to(args.device)
    if args.model == 'resnet32':
        return resnet32(num_classes=num_classes).to(args.device)
    if args.model == "resnet34":
        return resnet34(num_classes=num_classes).to(args.device)
    if args.model in ['VGG', 'vgg19', 'vgg19_bn']:
        return vgg19_bn().to(args.device)
    if args.model in ['VGG11', 'vgg11']:
        try:
            from models.Nets import vgg11
            return vgg11().to(args.device)
        except Exception:
            return vgg19_bn().to(args.device)
    if args.model == 'cnn':
        return get_model('fmnist').to(args.device)
    raise ValueError(f"Unsupported model for attacker utils: {args.model}")


def layer_analysis_no_acc(model_param, args, mal_train_dataset, mal_val_dataset, threshold=0.8):
    model = _build_attack_model(args)
    model.load_state_dict(model_param)

    model_benign = copy.deepcopy(model)
    acc, _ = test(copy.deepcopy(model_benign), mal_train_dataset, args)
    min_acc = 93 if args.dataset in ['cifar'] else 90
    num_time = 0
    while acc < min_acc:
        benign_train(model_benign, mal_train_dataset, args)
        num_time += 1
        if num_time % 4 == 0:
            acc, _ = test(copy.deepcopy(model_benign), mal_train_dataset, args, False)
            model = model_benign
            if num_time > 30:
                if acc > 80:
                    break
                else:
                    return []

    model_malicious = copy.deepcopy(model)
    model_malicious.load_state_dict(model.state_dict())
    malicious_train(model_malicious, mal_train_dataset, args)
    _, back_acc = test(model_malicious, mal_val_dataset, args)

    good_weight = model_benign.state_dict()
    bad_weight = model_malicious.state_dict()
    temp_weight = copy.deepcopy(good_weight)
    if args.attack_layers is None:
        args.attack_layers = []
    for layer in args.attack_layers:
        temp_weight[layer] = bad_weight[layer]
    temp_model = copy.deepcopy(model_benign)
    temp_model.load_state_dict(temp_weight)
    _, test_model_backdoor = test(temp_model, mal_val_dataset, args)
    if test_model_backdoor > threshold * back_acc:
        print(test_model_backdoor, ">", threshold * back_acc, "SKIP")
        return args.attack_layers

    key_arr, value_arr = FLS(model_benign, model_malicious, back_acc, mal_val_dataset, args)
    threshold = args.tau
    attack_list = IKPR(key_arr, value_arr, model_benign, model_malicious, back_acc, mal_val_dataset, args, threshold=threshold)
    print("finish identification")
    return attack_list




def get_attacker_dataset(args, dataset_train=None, dataset_test=None):
    if args.local_dataset==1:
        print("use local malicious dataset")
        mal_train_dataset, mal_val_dataset = split_dataset(args.data)
        return mal_train_dataset, mal_val_dataset
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        if dataset_train is None:
            dataset_train = datasets.MNIST(
                '../data/mnist/', train=True, download=True, transform=trans_mnist)
        if dataset_test is None:
            dataset_test = datasets.MNIST(
                '../data/mnist/', train=False, download=True, transform=trans_mnist)
        if args.iid:
            client_proportion = mnist_iid(dataset_train, args.num_users)
        else:
            client_proportion = mnist_noniid(dataset_train, args.num_users)
    if args.dataset in ['cifar']:
        norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ds_cls = datasets.CIFAR10
        data_root = '../data/cifar'
        trans_cifar = transforms.Compose([transforms.ToTensor(), norm])
        if dataset_train is None:
            dataset_train = ds_cls(
                data_root, train=True, download=True, transform=trans_cifar)
        if dataset_test is None:
            dataset_test = ds_cls(
                data_root, train=False, download=True, transform=trans_cifar)
        if args.iid:
            client_proportion = np.load('./data/iid_cifar.npy', allow_pickle=True).item()
        else:
            client_proportion = np.load('./data/non_iid_cifar.npy', allow_pickle=True).item()
    elif args.dataset == "fashion_mnist":
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.2860], std=[0.3530])])
        if dataset_train is None:
            dataset_train = datasets.FashionMNIST(
                '../data/', train=True, download=True, transform=trans_mnist)
        if dataset_test is None:
            dataset_test = datasets.FashionMNIST(
                '../data/', train=False, download=True, transform=trans_mnist)
        if args.iid:
            client_proportion = np.load('./data/iid_fashion_mnist.npy', allow_pickle=True).item()
        else:
            client_proportion = np.load('./data/non_iid_fashion_mnist.npy', allow_pickle=True).item()
    elif args.dataset == "gtsrb":
        if dataset_train is None:
            raise ValueError(f"Attacker dataset for {args.dataset} requires dataset_train to be passed in.")
        if args.iid:
            client_proportion = cifar_iid(dataset_train, args.num_users)
        else:
            client_proportion = dirichlet_split(getattr(dataset_train, "targets"), args.num_users, alpha=args.p)

    data_list = []
    begin_pos = 0
    malicious_client_num = int(args.num_users * args.malicious)
    for i in range(begin_pos, begin_pos + malicious_client_num):
        data_list.extend(client_proportion[i])
    attacker_label = []
    for i in range(len(data_list)):
        attacker_label.append(dataset_train.targets[data_list[i]])
    attacker_label = np.array(attacker_label)
    client_dataset = []
    for i in range(len(data_list)):
        client_dataset.append(dataset_train[data_list[i]])
    mal_train_dataset, mal_val_dataset = split_dataset(client_dataset)
    return mal_train_dataset, mal_val_dataset


def split_dataset(dataset):
    num_dataset = len(dataset)
    # random
    data_distribute = np.random.permutation(num_dataset)
    malicious_dataset = []
    mal_val_dataset = []
    mal_train_dataset = []
    for i in range(num_dataset):
        malicious_dataset.append(dataset[data_distribute[i]])
        if i < num_dataset // 4:
            mal_val_dataset.append(dataset[data_distribute[i]])
        else:
            mal_train_dataset.append(dataset[data_distribute[i]])
    return mal_train_dataset, mal_val_dataset


def get_attack_layers_no_acc(model_param, args):
    mal_train_dataset, mal_val_dataset = get_attacker_dataset(args)
    return layer_analysis_no_acc(model_param, args, mal_train_dataset, mal_val_dataset)


def get_malicious_info(model_param, args, dataset_train=None, dataset_test=None,flag_test=False,trigger=None):

    mal_train_dataset, mal_val_dataset = get_attacker_dataset(args, dataset_train, dataset_test)

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
    if flag_test==False:
        key_arr, value_arr, back_acc, benign_model, malicious_model = get_key_value_bsr(model_param, args,
                                                         mal_train_dataset,
                                                         mal_val_dataset,flag_test=flag_test,trigger=trigger)
        malicious_info = {'key_arr': key_arr, 'value_arr': value_arr, 'malicious_model_BSR': back_acc,
                          'mal_val_dataset': mal_val_dataset, 'benign_model':benign_model, 'malicious_model':malicious_model}
    else:
        key_arr, value_arr, back_acc, benign_model, malicious_model,layer_activation = get_key_value_bsr(model_param, args,
                                                         mal_train_dataset,
                                                         mal_val_dataset,flag_test=flag_test,trigger=trigger)
        malicious_info = {'key_arr': key_arr, 'value_arr': value_arr, 'malicious_model_BSR': back_acc,
                          'mal_val_dataset': mal_val_dataset, 'benign_model':benign_model, 'malicious_model':malicious_model,'layer_activation':layer_activation}
    return malicious_info


def get_malicious_info_local(local_benign_model, local_malicious_model, args, dataset_train=None, dataset_test=None):
    mal_train_dataset, mal_val_dataset = get_attacker_dataset(args, dataset_train, dataset_test)
    key_arr, value_arr, back_acc, benign_model, malicious_model = get_key_value_bsr_local(local_benign_model,local_malicious_model, args,
                                                     mal_val_dataset)
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
    malicious_info = {'key_arr': key_arr, 'value_arr': value_arr, 'malicious_model_BSR': back_acc,
                      'mal_val_dataset': mal_val_dataset, 'benign_model':benign_model, 'malicious_model':malicious_model}
    return malicious_info


def get_key_value_bsr(model_param, args, mal_train_dataset, mal_val_dataset,flag_test,trigger=None):
    model = _build_attack_model(args)
    param1 = model_param
    model.load_state_dict(param1)

    model_benign = copy.deepcopy(model)
    acc, backdoor = test(copy.deepcopy(model_benign), mal_train_dataset, args)
    if args.dataset in ['cifar']:
        min_acc = 93
    else:
        min_acc = 90
    num_time = 0
    while (acc < min_acc):
        benign_train(model_benign, mal_train_dataset, args)
        num_time += 1
        if num_time % 4 == 0:
            acc, _ = test(copy.deepcopy(model_benign), mal_train_dataset, args, False)
            model = model_benign
            if num_time > 30:
                if acc > 80:
                    break

    model_malicious = copy.deepcopy(model)
    model_malicious.load_state_dict(model.state_dict())
    malicious_train(model_malicious, mal_train_dataset, args,trigger=trigger)
    acc, back_acc = test(model_malicious, mal_val_dataset, args)
    key_arr, value_arr = FLS(model_benign, model_malicious, back_acc, mal_val_dataset, args)
    if flag_test==False:
        return key_arr, value_arr, back_acc, model_benign.state_dict(), model_malicious.state_dict()
    else:
        layer_activation=analyze_critical_neurons_with_fls(model_benign, model_malicious, back_acc, mal_val_dataset, args)
        return key_arr, value_arr, back_acc, model_benign.state_dict(), model_malicious.state_dict(),layer_activation


def get_key_value_bsr_local(local_model_benign, local_malicious_model, args, mal_val_dataset):
    model_benign = copy.deepcopy(local_model_benign)
    model_malicious = copy.deepcopy(local_malicious_model)
    acc, back_acc = test(model_malicious, mal_val_dataset, args)
    key_arr, value_arr = FLS(model_benign, model_malicious, back_acc, mal_val_dataset, args)
    return key_arr, value_arr, back_acc, model_benign.state_dict(), model_malicious.state_dict()

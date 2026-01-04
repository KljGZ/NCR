import sys

from attack.AttackerUtils import IKPR_neurons_proportional_sorted
from aggregation.Fed import FedAvg
from aggregation.Update import LocalUpdate

sys.path.append('../')

from random import random
from utils.test import test_img
from models.Nets import ResNet18, vgg19_bn, vgg19, get_model
from torch.utils.data import DataLoader, Dataset
from utils.options import args_parser

import torch
from torchvision import datasets, transforms
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
from attack.MaliciousUpdate import LocalMaliciousUpdate


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


def malicious_train(model, dataset, args):
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
            bad_data[xx] = add_trigger(args, bad_data[xx])
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


# backward layer substitution
def IKPR(key_arr, value_arr, model_benign, model_malicious, BSR, mal_val_dataset, args, threshold=0.8):
    good_weight = model_benign.state_dict()
    bad_weight = model_malicious.state_dict()
    n = 1
    temp_BSR = 0
    attack_list = []
    np_key_arr = np.array(key_arr)
    net3 = copy.deepcopy(model_benign)
    while (temp_BSR < BSR * threshold and n <= len(key_arr)):
        minValueIdx = heapq.nsmallest(n, range(len(value_arr)), value_arr.__getitem__)
        attack_list = list(np_key_arr[minValueIdx])
        param = copy.deepcopy(good_weight)
        for layer in attack_list:
            param[layer] = bad_weight[layer]
        net3.load_state_dict(param)
        acc, _, temp_BSR = test_img(net3, mal_val_dataset, args, test_backdoor=True)
        n += 1
    return attack_list


def IKPR_weight(key_arr, value_arr, model_benign, model_malicious, BSR, mal_val_dataset, args, global_model, threshold=0.8):
    good_weight = model_benign
    bad_weight = model_malicious

    n = 1
    temp_BSR = 0
    attack_list = []
    np_key_arr = np.array(key_arr)
    net3 = copy.deepcopy(global_model)

    while (temp_BSR < BSR * threshold and n <= len(key_arr)):
        minValueIdx = heapq.nsmallest(n, range(len(value_arr)), value_arr.__getitem__)
        attack_list = list(np_key_arr[minValueIdx])
        param = copy.deepcopy(good_weight)
        for layer in attack_list:
            param[layer] = bad_weight[layer]
        net3.load_state_dict(param)
        acc, _, temp_BSR = test_img(net3, mal_val_dataset, args, test_backdoor=True)
        n += 1
    return attack_list


def get_key_value_bsr(model_param, args, mal_train_dataset, mal_val_dataset):
    if args.model == 'resnet':
        model = ResNet18().to(args.device)
    elif args.model == 'VGG':
        model = vgg19_bn().to(args.device)
    elif args.model == 'rlr_mnist':
        model = get_model('fmnist').to(args.device)
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
                else:
                    attack_list = []
                    return attack_list

    model_malicious = copy.deepcopy(model)
    model_malicious.load_state_dict(model.state_dict())
    malicious_train(model_malicious, mal_train_dataset, args)
    acc, back_acc = test(model_malicious, mal_val_dataset, args)
    key_arr, value_arr = FLS(model_benign, model_malicious, back_acc, mal_val_dataset, args)
    return key_arr, value_arr, back_acc


def adaptive_attack_analysis(benign_model_weight_list, malicious_model_weight, global_model, args, num_mal=1):
    # if malicious client is selected return True
    malicious_model_weight_list = [malicious_model_weight for i in range(num_mal)]
    if args.defence == 'flame':
        res = adaptive_attack_analysis_flame(benign_model_weight_list, malicious_model_weight_list, args)
        if len(res) == 0:
            return False
        else:
            return True
    if args.defence == 'krum' or args.defence == 'multikrum' or args.defence == 'fltrust' or args.defence == 'avg' or args.defence == 'fld' or args.defence == 'RLR' or args.defence != None:
        benign_update_list = []
        for i in range(len(benign_model_weight_list)):
            benign_update_list.append(get_update(benign_model_weight_list[i], copy.deepcopy(global_model.state_dict())))
        malicious_update_list = []
        malicious_update = get_update(malicious_model_weight, copy.deepcopy(global_model.state_dict()))
        for i in range(num_mal):
            malicious_update_list.append(malicious_update)
        res = adaptive_attack_analysis_krum(benign_update_list, malicious_update_list, args.k, args)
        if len(res) == 0:
            return False
        else:
            return True


def adaptive_attack_analysis_flame(benign_model_weight_list, malicious_model_weight_list, args):
    malicious_num = len(malicious_model_weight_list)
    malicious_model_weight_list.extend(benign_model_weight_list)
    model_list = malicious_model_weight_list
    selected_client = flame_analysis(model_list, args)
    # print("attacker line378 selected_client", selected_client)
    selected_malicious = []
    for i in range(malicious_num):
        if i in selected_client:
            selected_malicious.append(i)
    return selected_malicious


def adaptive_attack_analysis_krum(benign_update_list, malicious_update, k, args):
    malicious_num = len(malicious_update)
    malicious_update.extend(benign_update_list)
    log_dis = False
    if args.log_distance == True:
        log_dis = True
        args.log_distance = False
    if args.defence == 'krum'  or args.defence == 'fltrust' or args.defence == 'avg' or args.defence == 'fld' or args.defence=='RLR':
        selected_client = multi_krum(malicious_update, k, args)
    elif args.defence == 'multikrum' or args.defence != 'multikrum':
        selected_client = multi_krum(malicious_update, k, args, multi_k=True)

    print(f"         Malicious client indices {selected_client} flagged as attackers by the defense")
    if log_dis == True:
        args.log_distance = True
    if min(selected_client) < malicious_num:
        return selected_client
    else:
        return []


def adaptive_attack_analysis_fld(benign_model_update, crafted_model_update, old_update, hvp, args):
    benign_distance = torch.norm((old_update + hvp) - benign_model_update)
    benign_transf = 0.01/benign_distance
    malicious_distance = torch.norm((old_update + hvp) - crafted_model_update)
    malicious_score = malicious_distance * benign_transf
    if malicious_score< 0.0105 :
        return 1
    elif malicious_score>0.0095:
        return -1
    else:
        return 0


def gather_models_benign_trained(global_model, malicious_list, dict_users, args, dataset_train):
    w_updates = []
    w_locals = []
    for client in malicious_list:
        local = LocalUpdate(
            args=args, dataset=dataset_train, idxs=dict_users[client])
        w, loss, _ = local.train(
            net=copy.deepcopy(global_model).to(args.device))
        w_updates.append(get_update(w, global_model))
        w_locals.append(copy.deepcopy(w))
    return w_updates, w_locals


def adaptive_attack(benign_model_list, malicious_info, global_model, args, mode, num_mal,neuron_level=True,local=None,trigger=None):
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
    center_model = FedAvg(benign_model_list)
    attack_layer_info, num_layer = IKPR_neurons_proportional_sorted(
        key_arr=malicious_info['key_arr'],
        value_arr=malicious_info['value_arr'],
        model_benign=malicious_info['benign_model'],
        model_malicious=malicious_info['malicious_model'],
        mal_val_dataset=malicious_info['mal_val_dataset'],
        args=args,
        global_model=global_model,
        trigger=trigger,
    )


    if args.dataset in ["cifar"]:
        lambda_val = args.cifar_scale
    else:
        lambda_val = args.cnn_scale
    # print(f"------------------------attack_layer num is {num_layer},layer info num is {num_layer_info}-----------------------")
    if neuron_level == True:
        crafted_model = craft_model_neuron_level(
            center_model, malicious_info['local_malicious_model'], global_model, attack_layer_info,
            lambda_val)
        pass
    else:
        attack_layer_info = None
        attack_layer = IKPR_adaptive(
            malicious_info['key_arr'],
            malicious_info['value_arr'],
            center_model,
            malicious_info['local_malicious_model'],
            benign_model_list,
            global_model,
            args,
            num_mal,
            n=num_layer,
            attack_layer_info=attack_layer_info,
            neuron_level=neuron_level,
        )
        crafted_model = craft_model(
            center_model, malicious_info['local_malicious_model'], global_model, attack_layer,
            lambda_val)
    net_test = copy.deepcopy(global_model)
    net_test.load_state_dict(crafted_model)
    # # _, _, BSR=test_img(net_test,malicious_info['mal_val_dataset'], args, test_backdoor=True)
    # # net_test.load_state_dict(crafted_model1)
    # _, _, BSR1=test_img(net_test,malicious_info['mal_val_dataset'], args, test_backdoor=True)
    return crafted_model

def craft_model_neuron_level(benign_model, malicious_model, global_model, attack_layer_info, lambda_value):
    """Interpolate malicious parameters into selected neurons of benign model."""

    crafted_state = copy.deepcopy(global_model.state_dict())

    for layer_name, _ in crafted_state.items():


        if layer_name not in attack_layer_info :
            crafted_state[layer_name] = benign_model[layer_name]
            continue

        layer_info = attack_layer_info[layer_name]
        if layer_info['proportion']==0:
            crafted_state[layer_name] = benign_model[layer_name]
            continue
        selected_neurons = layer_info['selected_neurons']
        global_w = global_model.state_dict()[layer_name].clone()
        benign_w = benign_model[layer_name].clone()
        malicious_w = malicious_model[layer_name].clone()

        crafted_param = benign_w.clone()

        crafted_param[selected_neurons] += (malicious_w[selected_neurons] - global_w[selected_neurons]) * lambda_value + max(0,
                                                                                                (1 - lambda_value)) * (
                                            benign_w[selected_neurons] - global_w[selected_neurons])
        crafted_state[layer_name] = crafted_param

    return crafted_state

def flipping_attack_crafted_model(benign_model, malicious_model, global_model, attack_layer):
    crafted_model={}
    for key, var in global_model.state_dict().items():
        if key in attack_layer:
            crafted_model[key]=2*var-malicious_model[key]
        else:
            crafted_model[key]=benign_model[key]
    return crafted_model

# --------------------------------------------------modify----------------------------------
def craft_model(benign_model, malicious_model, global_model, attack_layer, lambda_value):
    crafted_model = copy.deepcopy(global_model.state_dict())
    benign_w = benign_model
    malicious_w = malicious_model
    global_w = global_model.state_dict()
    for layer, val in crafted_model.items():
        if layer in attack_layer:
            try:
                crafted_model[layer] += (malicious_w[layer] - global_w[layer]) * lambda_value + max(0, (1 - lambda_value)) * (
                    benign_w[layer] - global_w[layer])
            except:
                crafted_model[layer] = benign_w[layer]
        else:
            crafted_model[layer] = benign_w[layer]
    return crafted_model

def parameters_dict_to_vector_flt(net_dict) -> torch.Tensor:
    vec = []
    for key, param in net_dict.items():
        if key.split('.')[-1] == 'num_batches_tracked':
            continue
        vec.append(param.view(-1))
    return torch.cat(vec)


def cos_param(p1,p2):
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
    return cos(parameters_dict_to_vector_flt(p1),parameters_dict_to_vector_flt(p2))


def binary_search_lambda(benign_model_list, center_model, malicious_model, attack_layer, global_model, args, num_mal,
                         max_time=5, lambda_init = 1):
    
    lambda_val = lambda_init
    lambda_upper = 0
    lambda_lower = 0
    times = 0
    max_time = args.search_times
    print('attacker line 487 attack_layer:',attack_layer)

    while times < max_time:
        temp_model = craft_model(center_model, malicious_model, global_model, attack_layer, lambda_val)
        accept=adaptive_attack_analysis(benign_model_list, temp_model, global_model, args, num_mal)
        print("attacker line490 lambda_val:", lambda_val)
        if accept is True:
            if lambda_upper == 0:
                # scale up
                lambda_lower = lambda_val
                lambda_val *= 2
            else:
                lambda_lower = lambda_val
                lambda_val = (lambda_lower + lambda_upper) / 2
        else:
            lambda_upper = lambda_val
            lambda_val = (lambda_upper + lambda_lower) / 2
        times+=1
        if times == max_time and lambda_lower==0:
            lambda_val=1
    return lambda_val



def compute_layer_scores(value_arr, key_arr, attack_layer_info):

    if len(key_arr) != len(value_arr):
        raise ValueError("key_arr and value_arr must have the same length.")
    IKPR_values = []
    for layer in key_arr:
        if layer not in attack_layer_info:
            raise KeyError(f"Layer '{layer}' not found in attack_layer_info.")
        ikpr_value = attack_layer_info[layer]['IKPR_value']
        IKPR_values.append(ikpr_value)

    value_arr_np = np.array(value_arr, dtype=float)
    value_arr_np=-value_arr_np
    IKPR_values_np = np.array(IKPR_values, dtype=float)

    def min_max_normalize(arr):
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val - min_val == 0:
            return np.full_like(arr, 0.5)
        return (arr - min_val) / (max_val - min_val)

    normalized_value = min_max_normalize(value_arr_np)
    normalized_ikpr = min_max_normalize(IKPR_values_np)


    combined_scores = normalized_value + normalized_ikpr

    return combined_scores.tolist()
def IKPR_adaptive(key_arr, value_arr, model_benign_weight, model_malicious_weight, benign_model_list, global_model,
                  args, num_mal, n=0,attack_layer_info=None,neuron_level=False):
    good_weight = model_benign_weight
    bad_weight = model_malicious_weight
    combined_scores = compute_layer_scores(value_arr, key_arr, attack_layer_info)
    attack_list = []
    np_key_arr = np.array(key_arr)

    if n == 0:
        n = 1
        # increasing the number of attacking layers
        while n <= len(key_arr):
            minValueIdx = heapq.nlargest(n, range(len(combined_scores)), combined_scores.__getitem__)
            attack_list_temp = list(np_key_arr[minValueIdx])
            param = copy.deepcopy(good_weight)
            for layer in attack_list_temp:
                param[layer] = bad_weight[layer]
            crafted_model = craft_model(model_benign_weight, model_malicious_weight, global_model, attack_list_temp, lambda_value=1)
            if_malicious_selected=adaptive_attack_analysis(benign_model_list, crafted_model, global_model, args, num_mal)
            if if_malicious_selected == False:
                break
            else:
                attack_list = attack_list_temp
                n += 1
    else:
        #decrease step by step
        first = True
        while n > 0:
            new_attack_layer_info = {}
            minValueIdx = heapq.nlargest(n, range(len(combined_scores)), combined_scores.__getitem__)
            attack_list_temp = list(np_key_arr[minValueIdx])

            if neuron_level==True:
                for layer in attack_list_temp:
                    new_attack_layer_info[layer_str] = attack_layer_info[layer_str]

                crafted_model = craft_model_neuron_level(benign_model=model_benign_weight,
                                                                malicious_model=model_malicious_weight,
                                                                global_model=global_model,
                                                                attack_layer_info=new_attack_layer_info, lambda_value=1)
            else:
                crafted_model = craft_model(model_benign_weight, model_malicious_weight, global_model, attack_list_temp,
                                            lambda_value=1)
            if_malicious_selected=adaptive_attack_analysis(benign_model_list, crafted_model, global_model, args, num_mal)
            if first == True:
                attack_list = attack_list_temp
                first = False
            if if_malicious_selected == False:
                break
            else:
                attack_list = attack_list_temp
                if n-10<0:
                    n=0
                elif n-10>=0:
                    n=n-10
                elif n-10==-10:
                    break
                print(
                )
    if neuron_level==True:
        return  new_attack_layer_info
    else:
        return attack_list



def test_eq(m1, m2):
    for layer in m1:
        if m1[layer].equal(m2[layer]):
            continue
        else:
            return False
    return True


def lambda_adaptive(key_arr, value_arr, model_benign, model_malicious, benign_model_list, malicious_model, global_model,
                    malicious_model_BSR, mal_val_dataset, args):
    attack_list = IKPR(
        key_arr, value_arr, model_benign, model_malicious, malicious_model_BSR, mal_val_dataset, args, threshold=0.8
    )

    return attack_list


def attacker(list_mal_client, num_mal, attack_type, dataset_train, dataset_test, dict_users, net_glob, args, idx=None,trigger=None):
    num_mal_temp=0
    if args.ada_mode == 20:
        temp_attack_layers = args.attack_layers
    if idx == None:
        idx = random.choice(list_mal_client)
    w, loss, args.attack_layers = None, None, None
    fisher = None
    # craft attack model once
    if trigger is None and attack_type != "badnet" and args.Trojan==True:
        idxs_tro = {key: dict_users[key] for key in list_mal_client if key in dict_users}
        local1= LocalMaliciousUpdate(args=args, dataset=dataset_train, idxs=idxs_tro, order=idx, dataset_test=dataset_test,trigger=None)
        trigger=local1.IMC_neurons(net= net_glob)


    local = LocalMaliciousUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], order=idx, dataset_test=dataset_test,trigger=trigger)
    if attack_type == "adaptive" :
        res = local.train(net=copy.deepcopy(net_glob).to(args.device), test_img=test_img)
        if len(res) == 2:
            loss, malicious_info = res
        else:
            loss, malicious_info, *_ = res
    else:
        res = local.train(net=copy.deepcopy(net_glob).to(args.device), test_img=test_img)
        if len(res) == 3:
            w, loss, _ = res
        else:
            w, loss = res
    print(f"---------------------Client:{idx} Start Atacking-----------------------------")
    if attack_type == "adaptive":
        num_benign_simulate = min(int(args.num_users * args.malicious), int(args.frac * args.num_users))
        
        if num_benign_simulate != int(args.frac * args.num_users):
            # decrease number of clients in simulation because number of malicious client are limited
            num_mal_temp = num_mal
            num_mal = int(args.num_users * args.malicious * args.malicious)

        num_benign_simulate -= num_mal
        benign_model_list = []
        for idx in range(num_benign_simulate):
            local = LocalUpdate(
                args=args, dataset=dataset_train, idxs=dict_users[idx])
            benign_w, loss, _ = local.train(
                net=copy.deepcopy(net_glob).to(args.device))
            benign_model_list.append(copy.deepcopy(benign_w))
        if args.ada_mode == 20:
            args.attack_layers = temp_attack_layers 
            w, args.attack_layers = adaptive_attack(benign_model_list, malicious_info, net_glob, args, args.ada_mode, num_mal,trigger=trigger)
        else:
            w = adaptive_attack(benign_model_list, malicious_info, net_glob, args, args.ada_mode, num_mal,trigger=trigger)
    if num_mal_temp>0:
        temp_w = [w for i in range(num_mal_temp)]
        w = temp_w
    elif num_mal > 0:
        temp_w = [w for i in range(num_mal)]
        w = temp_w
    
    # compute fisher on the resulting malicious model
    def _compute_fisher_from_weights(weight_dict):
        model_tmp = copy.deepcopy(net_glob).to(args.device)
        model_tmp.load_state_dict(weight_dict)
        return local.compute_fisher(model_tmp)

    if isinstance(w, list):
        fisher = []
        for wi in w:
            fisher.append(_compute_fisher_from_weights(wi))
    elif isinstance(w, dict):
        fisher = _compute_fisher_from_weights(w)
    else:
        fisher = None

    return w, loss, args.attack_layers,trigger, fisher

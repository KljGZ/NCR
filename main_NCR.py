#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from utils.test import test_img
from aggregation.Fed import FedAvg
from models.Nets import ResNet18, vgg19_bn, vgg19, get_model, vgg11
from models.Resnet import resnet32, resnet34
from aggregation.Update import LocalUpdate
from utils.info import print_exp_details, write_info_to_accfile, get_base_info
from utils.options import args_parser
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid
from defense.defense import fltrust, multi_krum, get_update, RLR, flame, get_update2, fld_distance, detection, detection1, parameters_dict_to_vector_flt, lbfgs_torch, layer_krum
from defense.AlignIns_defense import alignins_defense
from defense.Snowball import snowball_defense as snowball_full_defense
from defense.Scope_defense import scope_defense
from attack.Attacker import attacker
import torch
from torchvision import datasets, transforms
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib
import math
import os
import yaml
import datetime


matplotlib.use('Agg')


def write_file(filename, accu_list, back_list, args, analyse=False):
    write_info_to_accfile(filename, args)
    f = open(filename, "a")
    f.write("main_task_accuracy=")
    f.write(str(accu_list))
    f.write('\n')
    f.write("backdoor_accuracy=")
    f.write(str(back_list))
    if args.defence == "krum":
        krum_file = filename + "_krum_dis"
        torch.save(args.log_distance, krum_file)
    if analyse == True:
        need_length = len(accu_list) // 10
        acc = accu_list[-need_length:]
        back = back_list[-need_length:]
        best_acc = round(max(acc), 2)
        average_back = round(np.mean(back), 2)
        best_back = round(max(back), 2)
        f.write('\n')
        f.write('BBSR:')
        f.write(str(best_back))
        f.write('\n')
        f.write('ABSR:')
        f.write(str(average_back))
        f.write('\n')
        f.write('max acc:')
        f.write(str(best_acc))
        f.write('\n')
        f.close()
        return best_acc, average_back, best_back
    f.close()


def central_dataset_iid(dataset, dataset_size):
    all_idxs = [i for i in range(len(dataset))]
    central_dataset = set(np.random.choice(
        all_idxs, dataset_size, replace=False))
    return central_dataset


def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


if __name__ == '__main__':
    # parse args
    args = args_parser()
    if args.attack == 'NCR':
        args.attack = 'adaptive'  # adaptively control the number of attacking layers
    args.device = torch.device('cuda:{}'.format(
        args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    test_mkdir('./' + args.save)
    print_exp_details(args)

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST(
            '../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST(
            '../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'fashion_mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.2860], std=[0.3530])])
        dataset_train = datasets.FashionMNIST(
            '../data/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.FashionMNIST(
            '../data/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = np.load('./data/iid_fashion_mnist.npy', allow_pickle=True).item()
        else:
            dict_users = np.load('./data/non_iid_fashion_mnist.npy', allow_pickle=True).item()
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        dataset_train = datasets.CIFAR10(
            '../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10(
            '../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = np.load('./data/iid_cifar.npy', allow_pickle=True).item()
        else:
            # dict_users = np.load('./data/non_iid_cifar.npy', allow_pickle=True).item()
            dict_users = cifar_noniid([x[1] for x in dataset_train], args.num_users, 10, args.p)
            print('main_fed.py line 137 len(dict_users):', len(dict_users))
        if args.defence == 'alignins':
            dict_users = cifar_noniid([x[1] for x in dataset_train], args.num_users, 10, args.p)

    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    num_classes = 10
    if args.model == 'VGG' and args.dataset in ['cifar']:
        net_glob = vgg19_bn().to(args.device)
    elif args.model == 'VGG11' and args.dataset in ['cifar']:
        net_glob = vgg11().to(args.device)
    elif args.model == "resnet" and args.dataset in ['cifar']:
        net_glob = ResNet18(num_classes=num_classes).to(args.device)
    elif args.model == "resnet32" and args.dataset in ['cifar']:
        net_glob = resnet32(num_classes=num_classes).to(args.device)
    elif args.model == "resnet34" and args.dataset in ['cifar']:
        net_glob = resnet34(num_classes=num_classes).to(args.device)
    elif args.model == "cnn":
        net_glob = get_model('fmnist').to(args.device)
    else:
        exit('Error: unrecognized model')

    
        
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []

    if args.defence == 'fld':
        old_update_list = []
        weight_record = []
        update_record = []
        args.frac = 1
        malicious_score = torch.zeros((1, args.num_users))

    if math.isclose(args.malicious, 0):
        backdoor_begin_acc = 100
    else:
        backdoor_begin_acc = args.attack_begin  # overtake backdoor_begin_acc then attack
    central_dataset = central_dataset_iid(dataset_test, args.server_dataset)  # get root dataset for FLTrust
    base_info = get_base_info(args)
    filename = './' + args.save + '/accuracy_file_{}.txt'.format(base_info)  # log hyperparameters
    # record core hyperparameters at start
    write_info_to_accfile(filename, args)

    if args.init != 'None':  # continue from checkpoint
        param = torch.load(args.init)
        net_glob.load_state_dict(param)
        print("load init model")

    val_acc_list = [0.0001]  # Acc list
    backdoor_acculist = [0]  # BSR list

    args.attack_layers = []  # keep LSA

    if args.log_distance == True:
        args.krum_distance = []
        args.krum_layer_distance = []
    malicious_pool = int(args.num_users * args.malicious)
    malicious_list = [i for i in range(malicious_pool)]  # list of the index of malicious clients
    benign_list = [i for i in range(malicious_pool, args.num_users)]

    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    trigger=None
    for iter in range(args.epochs):
        # per-round state (for lr decay, schedulers)
        args.current_round = iter
        # reset trigger each round to allow IMC/TrojanNN re-optimization per round (state persists in args.imc_state)
        trigger = None
        loss_locals = []
        fisher_locals = []
        if not args.all_clients:
            w_locals = []
            w_updates = []
        m = max(int(args.frac * args.num_users), 1)  # target number of clients in each round
        selected_for_agg_ids = None  # track which clients are finally aggregated (by id)
        # decide per-round attack count before sampling clients
        if backdoor_begin_acc < val_acc_list[-1]:  # start attack only when Acc overtakes backdoor_begin_acc
            backdoor_begin_acc = 0
            attack_number = args.malicious_per_round if getattr(args, "malicious_per_round", -1) >= 0 else int(args.malicious * m)
        else:
            attack_number = 0
            
        if args.scaling_attack_round != 1:
            # scaling attack begin 100-th round and perform each args.attack_round round
            if iter > 100 and iter%args.scaling_attack_round == 0:
                attack_number = attack_number
            else:
                attack_number = 0
        # sample malicious clients from the global malicious pool; keep benign count consistent with m
        mal_sample_size = min(attack_number, len(malicious_list))
        round_attack_budget = mal_sample_size  # actual malicious uploads this round

        # client sampling
        if args.defence == 'fld':
            idxs_users = np.arange(args.num_users)
            if iter == 350:  # decrease the lr in the specific round to improve Acc
                args.lr *= 0.1
        else:
            # two-stage sampling: malicious first, then benign
            if mal_sample_size > 0:
                mal_sample = np.random.choice(malicious_list, mal_sample_size, replace=False)
            else:
                mal_sample = np.array([], dtype=int)
            benign_need = max(m - mal_sample_size, 0)
            if benign_need > 0:
                replace_flag = False if benign_need <= len(benign_list) else True
                ben_sample = np.random.choice(benign_list, benign_need, replace=replace_flag)
            else:
                ben_sample = np.array([], dtype=int)
            idxs_users = np.concatenate((mal_sample, ben_sample))  # malicious IDs come first by design
        mal_weight=[]
        mal_loss=[]
        round_attack_flags = []
        round_client_ids = [int(x) for x in list(idxs_users)]
        upload_records = []  # (client_id, is_malicious, weight_dict) in this round order
        print(f"[Round {iter}] sampled clients (ordered): {round_client_ids}")
        attack_number = round_attack_budget  # countdown during the per-client loop
        for num_turn, idx in enumerate(idxs_users):
            if attack_number > 0:  # upload models for malicious clients
                round_attack_flags.append(True)
                args.iter = iter
                if args.defence == 'fld':
                    args.old_update_list = old_update_list[0:round_attack_budget]
                    m_idx = idx
                else:
                    # make attacker index consistent with the sampled malicious client id for this round
                    # (DBA attack will override idx internally as designed)
                    m_idx = int(idx)
                mal_weight, loss, args.attack_layers ,trigger, mal_fisher= attacker(malicious_list, attack_number, args.attack, dataset_train, dataset_test, dict_users, net_glob, args, idx = m_idx,trigger=trigger)
                attack_number -= 1
                if isinstance(mal_weight, list):
                    w = mal_weight[0]
                    mf = mal_fisher[0] if isinstance(mal_fisher, list) else mal_fisher
                else:
                    w = mal_weight
                    mf = mal_fisher
                if getattr(args, "debug", 0):
                    print(f"[Attack] round {iter} client {idx} malicious upload; remaining this round {attack_number}")
            else:  # upload models for benign clients
                round_attack_flags.append(False)
                local = LocalUpdate(
                    args=args, dataset=dataset_train, idxs=dict_users[idx])
                w, loss, mf = local.train(
                    net=copy.deepcopy(net_glob).to(args.device))
                if getattr(args, "debug", 0):
                    print(f"[Benign] round {iter} client {idx} upload")
            if args.defence == 'fld':
                w_updates.append(get_update2(w, w_glob)) # ignore num_batches_tracked, running_mean, running_var
            else:
                w_updates.append(get_update(w, w_glob))
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            upload_records.append((int(idx), round_attack_flags[-1], copy.deepcopy(w)))
            loss_locals.append(copy.deepcopy(loss))
            fisher_locals.append(mf)

        # optional: evaluate malicious uploads' backdoor accuracy before aggregation
        if getattr(args, "eval_pre_agg", 0):
            pre_back_accs = []
            eval_net = copy.deepcopy(net_glob).to(args.device)
            for cid, is_attack, w_state in upload_records:
                if not is_attack:
                    continue
                eval_net.load_state_dict(w_state)
                _, _, back_acc_tmp = test_img(eval_net, dataset_test, args, test_backdoor=True)
                pre_back_accs.append((cid, back_acc_tmp))
            if pre_back_accs:
                print(f"[Round {iter}] pre-agg malicious backdoor acc:", pre_back_accs)
            else:
                print(f"[Round {iter}] pre-agg malicious backdoor acc: none (no malicious uploads)")

        if args.defence == 'avg':  # no defence
            w_glob = FedAvg(w_locals)
            selected_for_agg_ids = round_client_ids
        elif args.defence == 'krum':  # single krum
            selected_client = multi_krum(w_updates, 1, args)
            w_glob = w_locals[selected_client[0]]
            selected_for_agg_ids = [round_client_ids[selected_client[0]]]
        elif args.defence == 'multikrum':
            selected_client = multi_krum(w_updates, args.k, args, multi_k=True)
            w_glob = FedAvg([w_locals[x] for x in selected_client])
            selected_for_agg_ids = [round_client_ids[x] for x in selected_client]
        elif args.defence == 'RLR':
            w_glob = RLR(copy.deepcopy(net_glob), w_updates, args)
            selected_for_agg_ids = round_client_ids
        elif args.defence == 'fltrust':
            local = LocalUpdate(
                args=args, dataset=dataset_test, idxs=central_dataset)
            fltrust_norm, loss, _ = local.train(
                net=copy.deepcopy(net_glob).to(args.device))
            fltrust_norm = get_update(fltrust_norm, w_glob)
            w_glob = fltrust(w_updates, fltrust_norm, w_glob, args)
            selected_for_agg_ids = round_client_ids
        elif args.defence == 'flame':
            w_glob = flame(w_locals, w_updates, w_glob, args, debug=args.debug)
            selected_for_agg_ids = round_client_ids
        elif args.defence == 'alignins':
            # align client data sizes with this round's w_updates order (idxs_users iteration order)
            try:
                client_lens = [len(dict_users[idx]) for idx in idxs_users]
            except Exception:
                client_lens = None
            w_glob = alignins_defense(
                w_updates,
                w_glob,
                args,
                client_lens=client_lens,
                client_ids=round_client_ids,
                attack_flags=round_attack_flags,
            )
            selected_for_agg_ids = getattr(args, "alignins_selected_client_ids", None)
            if selected_for_agg_ids is None:
                idx_sel = getattr(args, "alignins_selected_indices", None)
                if idx_sel is not None:
                    selected_for_agg_ids = [round_client_ids[i] for i in idx_sel]
        elif args.defence == 'snowball':
            try:
                client_lens = [len(dict_users[idx]) for idx in idxs_users]
            except Exception:
                client_lens = None
            w_glob = snowball_full_defense(
                w_updates=w_updates,
                w_glob=w_glob,
                args=args,
                client_lens=client_lens,
                cur_round=iter + 1,
            )
            selected_for_agg_ids = round_client_ids
        elif args.defence == 'scope':
            try:
                client_lens = [len(dict_users[idx]) for idx in idxs_users]
            except Exception:
                client_lens = None
            w_glob = scope_defense(
                w_locals=w_locals,
                w_glob=w_glob,
                net_template=copy.deepcopy(net_glob).to(args.device),
                args=args,
                client_lens=client_lens,
            )
            selected_for_agg_ids = round_client_ids
        elif args.defence == 'flip':
            selected_client = multi_krum(w_updates, args.k, args, multi_k=True)
            w_glob = FedAvg([w_locals[x] for x in selected_client])
            selected_for_agg_ids = [round_client_ids[x] for x in selected_client]
        elif args.defence == 'flip_multikrum':
            selected_client = multi_krum(w_updates, args.k, args, multi_k=True)
            w_glob = FedAvg([w_locals[x] for x in selected_client])
            selected_for_agg_ids = [round_client_ids[x] for x in selected_client]
        elif args.defence == 'layer_krum':
            w_glob_update = layer_krum(w_updates, args.k, args, multi_k=True)
            for key, val in w_glob.items():
                w_glob[key] += w_glob_update[key]
            selected_for_agg_ids = round_client_ids
        elif args.defence == 'fld':
            # ignore key.split('.')[-1] == 'num_batches_tracked' or key.split('.')[-1] == 'running_mean' or key.split('.')[-1] == 'running_var'
            N = 5
            args.N = N
            weight = parameters_dict_to_vector_flt(w_glob)
            local_update_list = []
            for local in w_updates:
                local_update_list.append(-1*parameters_dict_to_vector_flt(local).cpu()) # change to 1 dimension
                
            if iter > N+1:
                args.hvp = lbfgs_torch(args, weight_record, update_record, weight - last_weight)
                hvp = args.hvp

                attack_number = round_attack_budget
                distance = fld_distance(old_update_list, local_update_list, net_glob, attack_number, hvp)
                distance = distance.view(1,-1)
                malicious_score = torch.cat((malicious_score, distance), dim=0)
                if malicious_score.shape[0] > N+1:
                    if detection1(np.sum(malicious_score[-N:].numpy(), axis=0)):
                        
                        label = detection(np.sum(malicious_score[-N:].numpy(), axis=0), round_attack_budget)
                    else:
                        label = np.ones(args.num_users)
                    selected_client = []
                    for client in range(args.num_users):
                        if label[client] == 1:
                            selected_client.append(client)
                    new_w_glob = FedAvg([w_locals[client] for client in selected_client])
                else:
                    new_w_glob = FedAvg(w_locals)  # avg
            else:
                hvp = None
                new_w_glob = FedAvg(w_locals)  # avg

            update = get_update2(w_glob, new_w_glob)  # w_t+1 = w_t - a*g_t => g_t = w_t - w_t+1 (a=1)
            update = parameters_dict_to_vector_flt(update)
            if iter > 0:
                weight_record.append(weight.cpu() - last_weight.cpu())
                update_record.append(update.cpu() - last_update.cpu())
            if iter > N:
                del weight_record[0]
                del update_record[0]
            last_weight = weight
            last_update = update
            old_update_list = local_update_list
            w_glob = new_w_glob
            selected_for_agg_ids = round_client_ids
        else:
            print("Wrong Defense Method")
            os._exit(0)

        # log per-round client flow
        uploaded_client_ids = round_client_ids  # all sampled clients upload an update
        if selected_for_agg_ids is None:
            selected_for_agg_ids = round_client_ids
        print(f"[Round {iter}] uploaded clients (ordered): {uploaded_client_ids}")
        print(f"[Round {iter}] aggregated clients: {selected_for_agg_ids}")

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

        if iter % 1 == 0:
            acc_test, _, back_acc = test_img(
                net_glob, dataset_test, args, test_backdoor=True)
            print("Main accuracy: {:.2f}".format(acc_test))
            print("Backdoor accuracy: {:.2f}".format(back_acc))
            val_acc_list.append(acc_test.item())

            backdoor_acculist.append(back_acc)
            write_file(filename, val_acc_list, backdoor_acculist, args)

    best_acc, absr, bbsr = write_file(filename, val_acc_list, backdoor_acculist, args, True)

    # plot loss curve
    plt.figure()
    plt.xlabel('communication')
    plt.ylabel('accu_rate')
    plt.plot(val_acc_list, label='main task(acc:' + str(best_acc) + '%)')
    plt.plot(backdoor_acculist, label='backdoor task(BBSR:' + str(bbsr) + '%, ABSR:' + str(absr) + '%)')
    plt.legend()
    title = base_info
    plt.title(title)
    plt.savefig('./' + args.save + '/' + title + '.pdf', format='pdf', bbox_inches='tight')

    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))
    
    torch.save(net_glob.state_dict(), './' + args.save + '/model' + '.pth')

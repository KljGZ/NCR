#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # save file 
    parser.add_argument('--save', type=str, default='save',
                        help="dic to save results (ending without /)")
    parser.add_argument('--init', type=str, default='None',
                        help="location of init model")
    # federated arguments
    parser.add_argument('--epochs', type=int, default=200,
                        help="rounds of training")
    parser.add_argument('--num_users', type=int,
                        default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1,
                        help="the fraction of clients: C")
    parser.add_argument('--malicious',type=float,default=0.1, help="proportion of mailicious clients")
    parser.add_argument('--malicious_per_round', type=int, default=2,
                        help="fixed number of malicious clients per round (-1 to use proportion-based)")
  
    #***** badnet labelflip layerattack updateflip get_weight  adaptive****
    parser.add_argument('--attack', type=str,
                        default='badnet', help='attack method')
    parser.add_argument('--ada_mode', type=int,
                        default=1, help='adaptive attack mode')
    parser.add_argument('--poison_frac', type=float, default=0.05,
                        help="fraction of dataset to corrupt for backdoor attack, 1.0 for layer attack")

    # *****local_ep = 3, local_bs=50, lr=0.1*******
    parser.add_argument('--local_ep', type=int, default=3,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=64,
                        help="local batch size: B")

    parser.add_argument('--bs', type=int, default=64, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01,
                        help="learning rate (legacy; client_lr preferred)")
    parser.add_argument('--client_lr', type=float, default=0.1,
                        help="client learning rate (SoDa-BNGuard default)")
    parser.add_argument('--lr_decay', type=float, default=1,
                        help="per-epoch exponential lr decay (SoDa-BNGuard default: 0.99)")

    # model arguments
    #*************************model******************************#
    # resnet cnn VGG mlp Mnist_2NN Mnist_CNN
    parser.add_argument('--model', type=str,
                        default='Mnist_CNN', help='model name')

    # other arguments
    #*************************dataset*******************************#
    # fashion_mnist mnist cifar cifar100
    parser.add_argument('--dataset', type=str,
                        default='mnist', help="name of dataset")
    
    parser.add_argument('--Trojan', type=bool,
                        default=True, help="Start Trojang trigger gernerate")
    
    #****0-avg, 1-fltrust 2-tr-mean 3-median 4-krum 5-muli_krum 6-RLR fltrust_bn fltrust_bn_lr****#
    parser.add_argument('--defence', type=str,
                        default='avg', help="strategy of defence")
    parser.add_argument('--k', type=int,
                        default=2, help="parameter of krum")
    # parser.add_argument('--iid', action='store_true',
    #                     help='whether i.i.d or not')
    parser.add_argument('--iid', type=int, default=1,
                        help='whether i.i.d or not')

 #************************atttack_label********************************#
    parser.add_argument('--attack_label', type=int, default=5,
                        help="trigger for which label")
    
    parser.add_argument('--single', type=int, default=0,
                        help="single shot or repeated")
    # attack_goal=-1 is all to one
    parser.add_argument('--attack_goal', type=int, default=-1,
                        help="trigger to which label")
    # --attack_begin 70 means accuracy is up to 70 then attack
    parser.add_argument('--attack_begin', type=int, default=0,
                        help="the accuracy begin to attack")
    # search times
    parser.add_argument('--search_times', type=int, default=20,
                        help="binary search times")
    
    parser.add_argument('--gpu', type=int, default=0,
                        help="GPU ID, -1 for CPU")
    parser.add_argument('--robustLR_threshold', type=int, default=4, 
                        help="break ties when votes sum to 0")
    
    parser.add_argument('--server_dataset', type=int,default=200,help="number of dataset in server")
    
    parser.add_argument('--server_lr', type=float,default=1,help="number of dataset in server using in fltrust")
    
    
    parser.add_argument('--momentum', type=float, default=0.9,
                        help="SGD momentum (SoDa-BNGuard default: 0.0)")
    parser.add_argument('--wd', type=float, default=0, help="SGD weight decay (SoDa-BNGuard default: 1e-4)")
    
    
    parser.add_argument('--split', type=str, default='user',
                        help="train-test split type, user or sample")   
    #*********trigger info*********
    #  square  apple  watermark  
    parser.add_argument('--trigger', type=str, default='square',
                        help="Kind of trigger")  
    # mnist 28*28  cifar10 32*32
    parser.add_argument('--triggerX', type=int, default='20',
                        help="position of trigger x-aix") 
    parser.add_argument('--triggerY', type=int, default='20',
                        help="position of trigger y-aix")
    
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--wrong_mal', type=int, default=0)
    parser.add_argument('--right_ben', type=int, default=0)
    
    parser.add_argument('--mal_score', type=float, default=0)
    parser.add_argument('--ben_score', type=float, default=0)
    
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--noise', type=float, default=0.001)
    parser.add_argument('--all_clients', action='store_true',
                        help='aggregation over all clients')
    parser.add_argument('--tau', type=float, default=0.8,
                        help="threshold of LPA_ER")
    parser.add_argument('--eval_pre_agg', type=int, default=0,
                        help="1 to evaluate backdoor accuracy of malicious uploads before aggregation")
    parser.add_argument('--debug', type=int, default=0, help="log debug info or not")
    parser.add_argument('--local_dataset', type=int, default=1, help="use local dataset for layer identification")
    parser.add_argument('--debug_fld', type=int, default=0, help="#1 save, #2 load")
    parser.add_argument('--decrease', type=float, default=0.3, help="proportion of dropped layers in robust experiments (used in mode11)")
    parser.add_argument('--increase', type=float, default=0.3, help="proportion of added layers in robust experiments (used in mode12)")
    parser.add_argument('--mode10_tau', type=float, default=0.95, help="threshold of mode 10")
    parser.add_argument('--cnn_scale', type=float, default=0.5, help="scale of cnn")
    parser.add_argument('--cifar_scale', type=float, default=1, help="scale of larger model")
    
    parser.add_argument('--num_layer', type=int, default=3, help="fixed number of layer attacks")
    
    parser.add_argument('--num_identification', type=int, default=1, help="fixed number of round to identify")
    parser.add_argument('--beta', type=float, default=0.5, help="weight of regularization loss in distance awareness attacks")
    parser.add_argument('--log_distance', type=bool, default=False, help="output krum distance")
    parser.add_argument('--scaling_attack_round', type=int, default=1, help="rounds of attack implements")
    parser.add_argument('--scaling_param', type=float, default=5, help="scaling up how many times")
    parser.add_argument('--p', type=float, default=0.5, help="level of non-iid")

    parser.add_argument('--layer_threshold', type=float, default=1)
    parser.add_argument('--global_threshold', type=float, default=1)

    # AlignIns defence hyperparameters
    parser.add_argument('--alignins_sparsity', type=float, default=0.5, help="top-k ratio for AlignIns MPSA")
    parser.add_argument('--lambda_s', type=float, default=1.0, help="AlignIns threshold for MPSA")
    parser.add_argument('--lambda_c', type=float, default=1.0, help="AlignIns threshold for TDA")
    parser.add_argument('--debug_alignins', type=int, default=1, help="1: print AlignIns selected clients per round")

    # Snowball defence hyperparameters (full AAAI'24 version)
    parser.add_argument('--snowball_ct', type=int, default=4, help="Snowball: cluster threshold (ct)")
    parser.add_argument('--snowball_vt', type=float, default=0.5, help="Snowball: top-down target ratio (vt)")
    parser.add_argument('--snowball_v_step', type=float, default=0.05, help="Snowball: expansion step (v_step)")
    parser.add_argument('--snowball_vae_hidden', type=int, default=256, help="Snowball: VAE hidden dim")
    parser.add_argument('--snowball_vae_latent', type=int, default=64, help="Snowball: VAE latent dim")
    parser.add_argument('--snowball_vae_initial', type=int, default=270, help="Snowball: VAE warmup epochs")
    parser.add_argument('--snowball_vae_tuning', type=int, default=30, help="Snowball: VAE tuning epochs")
    parser.add_argument('--snowball_layers', type=str, default="", help="Snowball: comma-separated layer name patterns to use")
    parser.add_argument('--snowball_warmup', type=int, default=-1, help="Snowball: override warmup rounds (-1 to use dataset defaults)")
    # MobileNetV2 width multiplier (default 1.0; set e.g., 0.3 for slim)
    parser.add_argument('--width_mult', type=float, default=1.0,
                        help="MobileNetV2 width multiplier (default 1.0; set smaller for slim models)")

    args = parser.parse_args()
    return args

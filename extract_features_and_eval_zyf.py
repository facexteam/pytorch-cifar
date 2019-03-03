#!/usr/bin/env python
# Train CIFAR10 with PyTorch.
# based on: https://github.com/kuangliu/pytorch-cifar
# maintainer: zhaoyafei (https://github.com/walkoncross, zhaoyafei0210@gmail.com)

from __future__ import print_function
import os
import os.path as osp
import argparse
import time
# import json

import numpy as np
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from models import *

from verification_eval.eval_roc_and_pr import eval_roc_and_pr


def add_arg_parser():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--net', default='resnet20_cifar10_nofc',
                        type=str, help='network architeture')
    parser.add_argument('--emb-size', default=32,
                        type=int, help='embedding size for some network architeture')
    parser.add_argument('--gpu-ids', default='0',
                        type=str, help='which GPUs to train on, set to "0,1,2" to use multiple GPUs')
    parser.add_argument('--checkpoint', type=str,
                        help='path to checkpoint')
    parser.add_argument('--test-bs', type=int, default=200,
                        help='the batch size for test data')
    parser.add_argument('--test-dw', type=int, default=4,
                        help='workers to load test data')
    parser.add_argument('--cifar-dir', type=str, default='./data',
                        help='path to save downloaded cifar dataset')
    parser.add_argument('--save-dir', type=str, default='./rlt-features-verif',
                        help='path to save features and verification results')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='name of dataset')
    parser.add_argument('--save-features', type=int, default=0,
                        help='whether to save features or not')
    parser.add_argument('--pairs-file', type=str, default='',
                        help='pairs file')
    return parser


def main():
    parser = add_arg_parser()
    args = parser.parse_args()
    pairs_filepairs = args.pairs_file

    dataset_name = args.dataset.lower()
    n_classes = 10

    if not pairs_file:
        pairs_file = './test_list/cifar100_test_pairs-same9990-diff9990_real_idx.txt'

    if dataset_name == 'cifar100':
        n_classes = 100
        if not pairs_file:
            pairs_file = './test_list/cifar100_test_pairs-same9900-diff9900_real_idx.txt'

    if 10000 % args.test_bs != 0:
        print("===> Must have: (10000 %% args.test_bs == 0)")
        return

    if args.test_bs % args.test_dw != 0:
        print("===> Must have: (args.test_bs %% args.test_dw == 0)")
        return

    print('===> Settings: ')
    print(args)

    gpu_ids = []
    if ',' in args.gpu_ids:
        gpu_ids = [int(id) for id in args.gpu_ids.split(',')]
    else:
        gpu_ids = [int(args.gpu_ids)]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if len(gpu_ids) == 1 and device == 'cuda':
        device = 'cuda:'+str(gpu_ids[0])

    last_epoch = -1  # start from epoch 0 or last checkpoint epoch

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    do_download = True

    if dataset_name == 'cifar100':
        if osp.exists(osp.join(args.cifar_dir, 'cifar-100-python.tar.gz')):
            print('cifar100 has already been downloaded to ', args.cifar_dir)
            do_download = False

        testset = torchvision.datasets.CIFAR100(
            root=args.cifar_dir, train=False, download=do_download, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.test_bs, shuffle=False, num_workers=args.test_dw)
    else:
        if osp.exists(osp.join(args.cifar_dir, 'cifar-10-python.tar.gz')):
            print('cifar10 has already been downloaded to ', args.cifar_dir)
            do_download = False

        testset = torchvision.datasets.CIFAR10(
            root=args.cifar_dir, train=False, download=do_download, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.test_bs, shuffle=False, num_workers=args.test_dw)

    # classes = ('plane', 'car', 'bird', 'cat', 'deer',
    #            'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    net_name = args.net.lower()
    print('==> Building model..')

    # if net_name == 'VGG19'.lower():
    #     net = VGG('VGG19')
    # elif net_name == 'ResNet18'.lower():
    #     net = ResNet18()
    # elif net_name == 'PreActResNet18'.lower():
    #     net = PreActResNet18()
    # elif net_name == 'GoogLeNet'.lower():
    #     net = GoogLeNet()
    # elif net_name == 'DenseNet121'.lower():
    #     net = DenseNet121()
    # elif net_name == 'ResNeXt29_2x64d'.lower():
    #     net = ResNeXt29_2x64d()
    # elif net_name == 'MobileNet'.lower():
    #     net = MobileNet()
    # elif net_name == 'MobileNetV2'.lower():
    #     net = MobileNetV2()
    # elif net_name == 'DPN92'.lower():
    #     net = DPN92()
    # elif net_name == 'ShuffleNetG2'.lower():
    #     net = ShuffleNetG2()
    # elif net_name == 'SENet18'.lower():
    #     net = SENet18()
    # elif net_name == 'ShuffleNetV2'.lower():
    #     net = ShuffleNetV2(1)
    # else:
    #     net = ResNet20_cifar10()
    if net_name == 'ResNet32_cifar10_nofc'.lower():
        print('===> Network: ResNet32_cifar10_nofc')
        net = ResNet32_cifar10_nofc()
    elif net_name == 'ResNet44_cifar10_nofc'.lower():
        print('===> Network: ResNet44_cifar10_nofc')
        net = ResNet44_cifar10_nofc()
    elif net_name == 'ResNet56_cifar10_nofc'.lower():
        print('===> Network: ResNet56_cifar10_nofc')
        net = ResNet56_cifar10_nofc()
    elif net_name == 'ResNet110_cifar10_nofc'.lower():
        print('===> Network: ResNet110_cifar10_nofc')
        net = ResNet110_cifar10_nofc()
    elif net_name == 'ResNet20_cifar10_fc'.lower():
        print('===> Network: ResNet20_cifar10_fc')
        net = ResNet20_cifar10_fc(args.emb_size)
    elif net_name == 'ResNet32_cifar10_fc'.lower():
        print('===> Network: ResNet32_cifar10_fc')
        net = ResNet32_cifar10_fc(args.emb_size)
    elif net_name == 'ResNet44_cifar10_fc'.lower():
        print('===> Network: ResNet44_cifar10_fc')
        net = ResNet44_cifar10_fc(args.emb_size)
    elif net_name == 'ResNet56_cifar10_fc'.lower():
        print('===> Network: ResNet56_cifar10_fc')
        net = ResNet56_cifar10_fc(args.emb_size)
    elif net_name == 'ResNet110_cifar10_fc'.lower():
        print('===> Network: ResNet110_cifar10_fc')
        net = ResNet110_cifar10_fc(args.emb_size)
    else:
        print('===> Network: ResNet20_cifar10_nofc')
        net = ResNet20_cifar10_nofc()

    # loss_type = args.loss_type.lower()

    # if loss_type == 'asoftmax':
    #     print('===> Using Scaled/Normalized A-softmax loss')
    #     net = LargeMarginModule_ScaledASoftmax(
    #         net, n_classes, args.loss_scale,
    #         args.loss_m, args.loss_lambda)
    # elif loss_type == 'arcface':
    #     print('===> Using arcface loss')
    #     net = LargeMarginModule_arcface(
    #         net, n_classes, args.loss_scale, args.loss_m)
    # elif loss_type.startswith('spa'):
    #     if loss_type == 'spav2':
    #         print('===> Using spav2 Softmax loss')
    #         net = SpaSoftmax_v2(net, n_classes, args.loss_scale,
    #                             args.loss_m, args.loss_b)
    #     elif loss_type == 'spav3':
    #         print('===> Using spav3 Softmax loss')
    #         net = SpaSoftmax_v3(net, n_classes, args.loss_scale,
    #                             args.loss_m, args.loss_n,
    #                             args.loss_b)
    #     elif loss_type == 'spav4':
    #         print('===> Using spav4 Softmax loss')
    #         net = SpaSoftmax_v4(net, n_classes, args.loss_scale,
    #                             args.loss_m, args.loss_n,
    #                             args.loss_b)
    #     elif loss_type == 'spav5':
    #         print('===> Using spav5 Softmax loss')
    #         net = SpaSoftmax_v5(net, n_classes, args.loss_scale,
    #                             args.loss_m)
    #     elif loss_type == 'spav6':
    #         print('===> Using spav6 Softmax loss')
    #         net = SpaSoftmax_v6(net, n_classes, args.loss_scale,
    #                             args.loss_m, args.loss_n)
    #     else:
    #         print('===> Using SPA Softmax loss')
    #         net = SpaSoftmax(net, n_classes, args.loss_scale,
    #                          args.loss_m, args.loss_b)
    # else:  # cosface
    #     print('===> Using cosface loss')
    #     net = LargeMarginModule_cosface(
    #         net, n_classes, args.loss_scale, args.loss_m)

    print('\n===> net params: ', net)

    if device.startswith('cuda'):
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, device_ids=gpu_ids)
            cudnn.benchmark = True
        else:
            net = net.to(device)

    last_epoch = -1

    checkpoint = None
    if not args.checkpoint:
        args.checkpoint = args.save_dir

    ckpt_list = []

    if osp.isdir(args.checkpoint):
        # epoch = -1
        # ckpt = ''
        best_ckpt = ''

        ckpt_dict = {}
        epoch_list = []

        dir_list = os.listdir(args.checkpoint)

        # # try to find checkpoint with best ACC
        # for fn in dir_list:
        #     if fn.endswith('.t7') and 'best' in fn:
        #         best_ckpt = fn
        #         ckpt_list.append(best_ckpt)
        #         break

        # try to find the checkpoint for the last epoch
        for fn in dir_list:
            if not fn.endswith('.t7'):
                continue

            splits = fn.rsplit('-', 2)
            if splits[1] == 'best':  # checkpoint with best ACC
                best_ckpt = fn
                # ckpt_list.append(best_ckpt)
                continue

            t_epoch = int(splits[1])

            epoch_list.append(t_epoch)
            ckpt_dict[str(t_epoch)] = fn

            # if t_epoch > epoch:
            #     epoch = t_epoch
            #     ckpt = fn

        if len(epoch_list) > 0:
            epoch_list.sort(reverse=True)
            for i in epoch_list:
                ckpt_list.append(ckpt_dict[str(i)])

        print(
            "===> Will try to load checkpoint from (one by one, until success): \n", ckpt_list)

        # # if not found,  use model with best acc if available
        # if not ckpt and best_ckpt:
        #     ckpt = best_ckpt

        if len(ckpt_list) > 0:
            for ckpt in ckpt_list:  # try to load available/valid checkpoint
                ckpt = osp.join(args.checkpoint, ckpt)
                print('==> Try to load checkpoint: ', ckpt)
                try:
                    checkpoint = torch.load(ckpt, map_location=device)
                    break
                except Exception as e:
                    checkpoint = None
                    print('Failed to load')
                    print('Exception: \n', str(e))
    elif osp.isfile(args.checkpoint):
        # Load checkpoint.
        print('==> Try to load checkpoint: ', args.checkpoint)
        checkpoint = torch.load(
            args.checkpoint, map_location=device)
    else:
        print("===> Checkpoint is not a valid file/folder: ",
              args.checkpoint)
        print("===> Exit")
        return

    if checkpoint:
        print('===> loaded checkpoint:')
        # print(type(checkpoint['net'])) # OrderedDict
        # print(len(checkpoint['net']))
        checkpoint['net'].pop('linear.weight')
        checkpoint['net'].pop('linear.bias')
        # print(len(checkpoint['net']))

        # print(checkpoint['net'])
        net.load_state_dict(checkpoint['net'])
        # best_acc = checkpoint['acc']
        # last_epoch = checkpoint['epoch']
        print("===> Succeeded to load checkpoint")
        # return
    else:
        print("===> No available checkpoint loaded")
        print("===> Exit")
        return

    if not osp.exists(args.save_dir):
        os.makedirs(args.save_dir)

    net.eval()

    total_features = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            print('===> batch #%04d' % batch_idx)
            # inputs, targets = inputs.to(device), targets.to(device)
            # outputs, cos_theta = net(inputs, targets)
            inputs = inputs.to(device)
            features = net(inputs)
            print('features.shape=', features.shape)
            # print('features:\n', features)

            total_features.append(features)
            # if batch_idx+1 == 10:
            #     break

    total_features = np.vstack(total_features)
    print('===> total_features.shape=', total_features.shape)

    eval_roc_and_pr(total_features, pairs_file, args.save_dir)

    if args.save_features:
        save_fn = args.dataset + '_features.npy'
        save_fn = osp.join(args.save_dir, save_fn)
        np.save(save_fn, total_features)


if __name__ == '__main__':
    # augments = [
    #     '--net', 'resnet20_cifar10_nofc',
    #     '--test-bs', '20',
    #     '--test-dw', '4',
    #     '--checkpoint', './res20-cifar-best.t7',
    #     '--cifar-dir', './data',
    #     '--dataset', 'cifar10',
    # ]

    # sys.argv.extend(augments)
    main()

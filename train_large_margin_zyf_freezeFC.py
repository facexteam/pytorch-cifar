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

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from models import *

from models.large_margin_module import LargeMarginModule_cosface, LargeMarginModule_ScaledASoftmax, LargeMarginModule_arcface
from models.spa_softmax import SpaSoftmax
from models.spa_softmax_v2 import SpaSoftmax_v2
from models.spa_softmax_v3 import SpaSoftmax_v3
from models.spa_softmax_v4 import SpaSoftmax_v4
from models.spa_softmax_v5 import SpaSoftmax_v5
from models.spa_softmax_v6 import SpaSoftmax_v6


def add_arg_parser():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--net', default='resnet20_cifar10_nofc',
                        type=str, help='network architeture')
    parser.add_argument('--emb-size', default=32,
                        type=int, help='embedding size for some network architeture')
    parser.add_argument('--gpu-ids', default='0',
                        type=str, help='which GPUs to train on, set to "0,1,2" to use multiple GPUs')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--resume-checkpoint', type=str,
                        help='path to resume checkpoint')
    parser.add_argument('--num-epochs', type=int, default=320,
                        help='max num of epochs')
    parser.add_argument('--lr-scheduler', default='step',
                        type=str, help='learning rate scheduler type: ["step", "cosine"]')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--lr-min', default=0.0, type=float,
                        help='minimum learning rate used in cosine lr')
    parser.add_argument('--lr-factor', type=float, default=0.1,
                        help='the ratio to reduce lr on each step')
    parser.add_argument('--lr-step-epochs', type=str, default='160,240',
                        help='the epochs to reduce the lr, e.g. 160,240')
    parser.add_argument('--mom', type=float, default=0.9,
                        help='momentum for sgd')
    parser.add_argument('--wd', type=float, default=1e-4,
                        help='weight decay for sgd')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='the batch size for train')
    parser.add_argument('--data-workers', type=int, default=4,
                        help='workers to load train data')
    parser.add_argument('--test-bs', type=int, default=200,
                        help='the batch size for test data')
    parser.add_argument('--test-dw', type=int, default=4,
                        help='workers to load test data')
    parser.add_argument('--model-prefix', type=str, default='ckpt',
                        help='model prefix')
    parser.add_argument('--cifar-dir', type=str, default='./data',
                        help='path to save downloaded cifar dataset')
    parser.add_argument('--save-dir', type=str, default='./checkpoints',
                        help='path to save checkpoints')
    parser.add_argument('--save-thresh', type=float, default=0.9,
                        help='save checkpoints with test acc>save_thresh')
    parser.add_argument('--no-progress-bar', dest='progress_bar', action='store_false',
                        help='whether to show progress bar')
    parser.add_argument('--loss-type', type=str, default='cosface',
                        help='loss type: ["a-softmax", "cosface", "arcface", "spa", "spav2", "spav3", "spav4", "spav5", "spav6"]')
    parser.add_argument('--loss-scale', type=float, default=32,
                        help='loss param: scale')
    parser.add_argument('--loss-m', type=float, default=0.35,
                        help='loss param: m')
    parser.add_argument('--loss-n', type=float, default=1,
                        help='loss param: n for some losses')
    parser.add_argument('--loss-b', type=float, default=1,
                        help='loss param: b')
    parser.add_argument('--loss-lambda', type=float, default=5,
                        help='loss param: lambda for A-softmax')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='name of dataset')
    return parser


def main():
    parser = add_arg_parser()
    args = parser.parse_args()

    # to make 260 to 256, for example
    args.batch_size = args.batch_size // args.data_workers * args.data_workers

    dataset_name = args.dataset.lower()
    n_classes = 10
    
    if dataset_name == 'cifar100':
        n_classes = 100

    if 10000 % args.test_bs != 0:
        print("===> Must have: (10000 %% args.test_bs == 0)")
        return

    if args.test_bs % args.test_dw != 0:
        print("===> Must have: (args.test_bs %% args.test_bs == 0)")
        return

    if args.progress_bar:
        from utils import progress_bar

    print('===> Train settings: ')
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
    step_epochs = [int(l) for l in args.lr_step_epochs.split(',')]
    best_acc = 0  # best test accuracy

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

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

        trainset = torchvision.datasets.CIFAR100(
            root=args.cifar_dir, train=True, download=do_download, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.data_workers)

        trainset2 = torchvision.datasets.CIFAR100(
            root=args.cifar_dir, train=True, download=do_download, transform=transform_test)
        trainloader_test = torch.utils.data.DataLoader(
            trainset2, batch_size=args.test_bs, shuffle=False, num_workers=args.test_dw)

        testset = torchvision.datasets.CIFAR100(
            root=args.cifar_dir, train=False, download=do_download, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.test_bs, shuffle=False, num_workers=args.test_dw)
    else:
        if osp.exists(osp.join(args.cifar_dir, 'cifar-10-python.tar.gz')):
            print('cifar10 has already been downloaded to ', args.cifar_dir)
            do_download = False

        trainset = torchvision.datasets.CIFAR10(
            root=args.cifar_dir, train=True, download=do_download, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.data_workers)

        trainset2 = torchvision.datasets.CIFAR10(
            root=args.cifar_dir, train=True, download=do_download, transform=transform_test)
        trainloader_test = torch.utils.data.DataLoader(
            trainset2, batch_size=args.test_bs, shuffle=False, num_workers=args.test_dw)

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

    loss_type = args.loss_type.lower()

    if loss_type == 'asoftmax':
        print('===> Using Scaled/Normalized A-softmax loss')
        net = LargeMarginModule_ScaledASoftmax(
            net, n_classes, args.loss_scale,
            args.loss_m, args.loss_lambda)
    elif loss_type == 'arcface':
        print('===> Using arcface loss')
        net = LargeMarginModule_arcface(
            net, n_classes, args.loss_scale, args.loss_m)
    elif loss_type.startswith('spa'):
        if loss_type == 'spav2':
            print('===> Using spav2 Softmax loss')
            net = SpaSoftmax_v2(net, n_classes, args.loss_scale,
                                args.loss_m, args.loss_b)
        elif loss_type == 'spav3':
            print('===> Using spav3 Softmax loss')
            net = SpaSoftmax_v3(net, n_classes, args.loss_scale,
                                args.loss_m, args.loss_n,
                                args.loss_b)
        elif loss_type == 'spav4':
            print('===> Using spav4 Softmax loss')
            net = SpaSoftmax_v4(net, n_classes, args.loss_scale,
                                args.loss_m, args.loss_n,
                                args.loss_b)
        elif loss_type == 'spav5':
            print('===> Using spav5 Softmax loss')
            net = SpaSoftmax_v5(net, n_classes, args.loss_scale,
                                args.loss_m)
        elif loss_type == 'spav6':
            print('===> Using spav6 Softmax loss')
            net = SpaSoftmax_v6(net, n_classes, args.loss_scale,
                                args.loss_m, args.loss_n)
        else:
            print('===> Using SPA Softmax loss')
            net = SpaSoftmax(net, n_classes, args.loss_scale,
                             args.loss_m, args.loss_b)
    else:  # cosface
        print('===> Using cosface loss')
        net = LargeMarginModule_cosface(
            net, n_classes, args.loss_scale, args.loss_m)

    print('\n===> net : ', net)
    # print('\n===> net params: ')
    # for param in net.parameters():
    #     print('\t', param)

    # print('\n===> net named_children: ')
    # for name, module in net.named_children():
    #     #print('{}: {}'.format(name, module))
    #     print('\t', name)

    if device.startswith('cuda'):
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, device_ids=gpu_ids)
            cudnn.benchmark = True
        else:
            net = net.to(device)

    last_epoch = -1

    if args.resume:
        checkpoint = None
        if not args.resume_checkpoint:
            args.resume_checkpoint = args.save_dir

        ckpt_list = []

        if osp.isdir(args.resume_checkpoint):
            # epoch = -1
            # ckpt = ''
            best_ckpt = ''

            ckpt_dict = {}
            epoch_list = []

            dir_list = os.listdir(args.resume_checkpoint)

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
                if splits[1]=='best':  # checkpoint with best ACC
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
                
            print("===> Will try to load checkpoint from (one by one, until success): \n", ckpt_list)
                
            # # if not found,  use model with best acc if available
            # if not ckpt and best_ckpt:
            #     ckpt = best_ckpt

            if len(ckpt_list) > 0:
                for ckpt in ckpt_list: # try to load available/valid checkpoint
                    ckpt = osp.join(args.resume_checkpoint, ckpt)
                    print('==> Try to load checkpoint: ', ckpt)
                    try:
                        checkpoint = torch.load(ckpt, map_location=device)
                        break
                    except Exception as e:
                        checkpoint = None
                        print('Failed to load')
                        print('Exception: \n', str(e))
        elif osp.isfile(args.resume_checkpoint):
            # Load checkpoint.
            print('==> Try to load checkpoint: ', args.resume_checkpoint)
            checkpoint = torch.load(
                args.resume_checkpoint, map_location=device)
        else:
            print("===> Resume checkpoint is not a valid file/folder: ", args.resume_checkpoint)
            print("===> Exit")
            return

        if checkpoint:
            net.load_state_dict(checkpoint['net'])
            best_acc = checkpoint['acc']
            last_epoch = checkpoint['epoch']
        else:
            last_epoch = -1
            args.resume = False
            print("\n===> Will train from scratch")
            # print("===> No available checkpoint loaded")
            # print("===> Exit")
            # return

    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=args.mom, weight_decay=args.wd)

    if args.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                         T_max=args.num_epochs,
                                                         eta_min=args.lr_min)
                                                         #  last_epoch=last_epoch
                                                        #  )
    else:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=step_epochs,
                                                   gamma=args.lr_factor)
                                                   #    last_epoch=last_epoch
                                                #    )

    if not osp.exists(args.save_dir):
        os.makedirs(args.save_dir)

    log_fn = osp.join(args.save_dir, 'train-log.txt')
    loss_fn = osp.join(args.save_dir, 'train-loss.txt')
    fc_fn = osp.join(args.save_dir, 'train-last-fc.txt')

    open_flag = 'w'
    if args.resume:
        open_flag = 'a+'

    fp_log = open(log_fn, open_flag)
    fp_log.write("===> TRAIN ARGS:\n")
    fp_log.write(str(args)+'\n')
    fp_log.write("===<\n")
    fp_log.close()

    fp_loss = open(loss_fn, open_flag)
    loss_log_format = '{epoch}\t{lr}\t{train_loss}\t{test_loss}\t{train_acc}\t{test_acc}\t{train_cos}\t{test_cos}\t{train_ang}\t{test_ang}\t{avg_fc_cos_max}\t{avg_fc_ang_min}'
    if last_epoch < 0:
        fp_loss.write(loss_log_format + '\n')
        fp_loss.flush()

    fp_fc = open(fc_fn, open_flag)
    fc_log_format = '{epoch}\t{avg_fc_cos_max}\t{avg_fc_ang_min}\t{fc_cos_max}\t{fc_ang_min}\t{fc_cos_mat}\t{fc_ang_mat}\t{fc_wt}'
    if last_epoch < 0:
        fp_fc.write(fc_log_format + '\n')
        fp_fc.flush()

    # Training
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        avg_loss = 0
        acc = 0

        cosine_sum = 0
        angle_sum = 0
        avg_cosine = 0
        avg_angle = 0

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)

            # print('\n===> In train()\n')
            # print('---> targets:', targets)
            # print('---> targets.shape:', targets.shape)

            optimizer.zero_grad()
            outputs, cos_theta = net(inputs, targets)

            # print('---> outputs:', outputs)
            # print('---> cos_theta:', cos_theta)

            loss = criterion(outputs, targets)
            loss.backward()
            net.linear.weight.requires_grad = False
            optimizer.step()
            net.linear.weight.requires_grad = True

            idx_mat = targets.reshape(-1, 1)
            # print('---> idx_mat:', idx_mat)
            # print('---> idx_mat.shape:', idx_mat.shape)

            cosine_mat = torch.gather(cos_theta, 1, idx_mat)
            cosine_mat = cosine_mat.clamp(-1, 1)
            # print('---> cosine_mat:', cosine_mat)
            # print('---> cosine_mat.shape:', cosine_mat.shape)
            # print('---> cosine_mat.max():', cosine_mat.max())
            # print('---> cosine_mat.min():', cosine_mat.min())
            # print('---> cosine_mat.mean():', cosine_mat.mean())

            cosine_sum += cosine_mat.mean().item()
            avg_cosine = cosine_sum / (batch_idx + 1)

            angle_mat = cosine_mat.acos()
            # print('---> angle_mat:', angle_mat)
            # print('---> angle_mat.shape:', angle_mat.shape)
            # print('---> angle_mat.max():', angle_mat.max())
            # print('---> angle_mat.min():', angle_mat.min())
            # print('---> angle_mat.mean():', angle_mat.mean())

            angle_sum += angle_mat.mean().item() * 180 / np.pi
            avg_angle = angle_sum / (batch_idx + 1)

            train_loss += loss.item()
            _, predicted = cos_theta.max(1)
            # print('---> predicted:', predicted)

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            avg_loss = train_loss / (batch_idx + 1)
            acc = float(correct)/total

            if args.progress_bar:
                progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Avg-Cos: %.3f | Avg-Angle(degree): %6.3f'
                             % (avg_loss, acc*100, correct, total, avg_cosine, avg_angle))

        return avg_loss, acc, avg_cosine, avg_angle

    def test(epoch, dataloader):
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        avg_loss = 0
        acc = 0

        cosine_sum = 0
        angle_sum = 0
        avg_cosine = 0
        avg_angle = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(device), targets.to(device)

                outputs, cos_theta = net(inputs, targets)

                loss = criterion(outputs, targets)

                idx_mat = targets.reshape(-1, 1)
                # print('---> idx_mat:', idx_mat)
                # print('---> idx_mat.shape:', idx_mat.shape)

                cosine_mat = torch.gather(cos_theta, 1, idx_mat)
                cosine_mat = cosine_mat.clamp(-1, 1)

                # print('---> cosine_mat:', cosine_mat)
                # print('---> cosine_mat.shape:', cosine_mat.shape)
                # print('---> cosine_mat.max():', cosine_mat.max())
                # print('---> cosine_mat.min():', cosine_mat.min())
                # print('---> cosine_mat.mean():', cosine_mat.mean())

                cosine_sum += cosine_mat.mean().item()
                avg_cosine = cosine_sum / (batch_idx + 1)

                angle_mat = cosine_mat.acos()
                # print('---> angle_mat:', angle_mat)
                # print('---> angle_mat.shape:', angle_mat.shape)
                # print('---> angle_mat.max():', angle_mat.max())
                # print('---> angle_mat.min():', angle_mat.min())
                # print('---> angle_mat.mean():', angle_mat.mean())

                angle_sum += angle_mat.mean().item() * 180 / np.pi
                avg_angle = angle_sum / (batch_idx + 1)

                test_loss += loss.item()
                _, predicted = cos_theta.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                avg_loss = test_loss / (batch_idx + 1)
                acc = float(correct)/total

                if args.progress_bar:
                    progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Avg-Cosine: %.3f | Avg-Angle(degree): %6.3f'
                                 % (avg_loss, acc*100, correct, total, avg_cosine, avg_angle))

        return avg_loss, acc, avg_cosine, avg_angle

    # for epoch in range(last_epoch, last_epoch+args.num_epochs):
    for epoch in range(0, args.num_epochs):
        scheduler.step()
        lr = scheduler.get_lr()
        if args.resume and epoch <= last_epoch: # reset lr scheduler
            print('\n---> skip epoch #%d: lr=%f' % (epoch, lr[0]))
            continue

        print('\n---> lr=', lr[0])

        train_loss, train_acc, train_cos, train_ang = train(epoch)

        train_loss, train_acc, train_cos, train_ang = test(
            epoch, trainloader_test)
        test_loss, test_acc, test_cos, test_ang = test(epoch, testloader)

        fc_wt = net.get_fc_weights()
        fc_wt_n = F.normalize(fc_wt, dim=1)
        fc_cos_mat = fc_wt_n.mm(fc_wt_n.t())
        fc_cos_mat = fc_cos_mat.clamp(-1, 1)

        fc_ang_mat = fc_cos_mat.acos() * 180 / np.pi

        # fc_cos_mat2 = fc_cos_mat - torch.diag(fc_cos_mat.diag())
        # remove diagnal elements
        fc_cos_mat2 = fc_cos_mat - \
            torch.eye(fc_cos_mat.shape[0], device=device)*10
        fc_cos_max, pos = fc_cos_mat2.max(dim=0)
        fc_ang_min = fc_ang_mat[pos].diag()

        avg_fc_cos_max = fc_cos_max.mean().item()
        avg_fc_ang_min = fc_ang_min.mean().item()

        # '{}\t{}\t{}\t{}\t{}\t{} \n'
        msg = loss_log_format.format(
            epoch=epoch, lr=lr[0],
            train_loss=train_loss, train_acc=train_acc,
            test_loss=test_loss, test_acc=test_acc,
            train_cos=train_cos, train_ang=train_ang,
            test_cos=test_cos, test_ang=test_ang,
            avg_fc_cos_max=avg_fc_cos_max,
            avg_fc_ang_min=avg_fc_ang_min)

        print('====>\n' + loss_log_format + '\n' + msg + '\n')
        fp_loss.write(msg+'\n')
        fp_loss.flush()

        # msg2 = fc_log_format.format(
        #     epoch=epoch,
        #     avg_fc_cos_max=avg_fc_cos_max,
        #     avg_fc_ang_min=avg_fc_ang_min,
        #     fc_cos_max=fc_cos_max.tolist(),
        #     fc_ang_min=fc_ang_min.tolist(),
        #     fc_cos_mat=0,
        #     fc_ang_mat=0,
        #     fc_wt=0)

        # print('====>\n' + fc_log_format + '\n' + msg2 + '\n')
        # # print('====>\nfc_wt.shape:', fc_wt.shape)
        # # print('====>\fc_cos_mat.shape:', fc_cos_mat.shape)

        msg3 = fc_log_format.format(
            epoch=epoch,
            avg_fc_cos_max=avg_fc_cos_max,
            avg_fc_ang_min=avg_fc_ang_min,
            fc_cos_max=fc_cos_max.tolist(),
            fc_ang_min=fc_ang_min.tolist(),
            fc_cos_mat=fc_cos_mat.tolist(),
            fc_ang_mat=fc_ang_mat.tolist(),
            fc_wt=fc_wt.tolist())

        fp_fc.write(msg3+'\n')
        fp_fc.flush()

        # Save checkpoint.
        # if test_acc >= args.save_thresh or epoch %:
        print('Saving..')
        state = {
            # 'net': net.state_dict(),
            'net': net.cpu().state_dict(),  # use cpu data to fix cuda OOM when saving
            'acc': test_acc,
            'epoch': epoch,
        }

        save_name = osp.join(args.save_dir, '%s-%04d-testacc%4.2f' %
                             (args.model_prefix, epoch, test_acc * 100))
        if test_acc > best_acc:
            best_acc = test_acc
            save_name += '_best'
        
        save_name += '.t7'
        # if test_acc > best_acc:
        time.sleep(10)
        torch.save(state, save_name)
        net.cuda(device)

        # # Save checkpoint.
        # if test_acc > best_acc:
        #     print('Saving..')
        #     state = {
        #         # 'net': net.state_dict(),
        #         'net': net.cpu().state_dict(),  # use cpu data to fix cuda OOM when saving
        #         'acc': test_acc,
        #         'epoch': epoch,
        #     }

        #     best_acc = test_acc

        #     if epoch > args.num_epochs/2:
        #         time.sleep(10)
        #         save_name = osp.join(args.save_dir, '%s-best-acc.t7' %
        #                              (args.model_prefix))
        #         torch.save(state, save_name)
                # net.cuda(device)

    # fp_log.close()
    fp_loss.close()
    fp_fc.close()


if __name__ == '__main__':
    main()


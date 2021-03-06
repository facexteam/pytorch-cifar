#!/usr/bin/env python
# Train CIFAR10 with PyTorch.
# based on: https://github.com/kuangliu/pytorch-cifar
# maintainer: zhaoyafei (https://github.com/walkoncross, zhaoyafei0210@gmail.com)

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import os.path as osp
import argparse

from models import *
from utils import progress_bar
import time


def add_arg_parser():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--net', default='resnet20_cifar',
                        type=str, help='network architeture')
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
    return parser


def main():
    parser = add_arg_parser()
    args = parser.parse_args()

    # to make 260 to 256, for example
    args.batch_size = args.batch_size // args.data_workers * args.data_workers

    if 10000 % args.test_bs != 0:
        print("===> Must have: (10000 %% args.test_bs == 0)")
        return

    if args.test_bs % args.test_dw != 0:
        print("===> Must have: (args.test_bs %% args.test_bs == 0)")
        return

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

    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
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
    if osp.exists(osp.join(args.cifar_dir, 'cifar-10-python.tar.gz')):
        print('cifar10 has already been downloaded to ', args.cifar_dir)
        do_download = False

    trainset = torchvision.datasets.CIFAR10(
        root=args.cifar_dir, train=True, download=do_download, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.data_workers)

    testset = torchvision.datasets.CIFAR10(
        root=args.cifar_dir, train=False, download=do_download, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_bs, shuffle=False, num_workers=args.test_dw)

    # classes = ('plane', 'car', 'bird', 'cat', 'deer',
    #            'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    net_name = args.net.lower()
    print('==> Building model..')

    if net_name == 'VGG19'.lower():
        net = VGG('VGG19')
    elif net_name == 'ResNet18'.lower():
        net = ResNet18()
    elif net_name == 'PreActResNet18'.lower():
        net = PreActResNet18()
    elif net_name == 'GoogLeNet'.lower():
        net = GoogLeNet()
    elif net_name == 'DenseNet121'.lower():
        net = DenseNet121()
    elif net_name == 'ResNeXt29_2x64d'.lower():
        net = ResNeXt29_2x64d()
    elif net_name == 'MobileNet'.lower():
        net = MobileNet()
    elif net_name == 'MobileNetV2'.lower():
        net = MobileNetV2()
    elif net_name == 'DPN92'.lower():
        net = DPN92()
    elif net_name == 'ShuffleNetG2'.lower():
        net = ShuffleNetG2()
    elif net_name == 'SENet18'.lower():
        net = SENet18()
    elif net_name == 'ShuffleNetV2'.lower():
        net = ShuffleNetV2(1)
    else:
        net = ResNet20_cifar10()

    if device.startswith('cuda'):
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, device_ids=gpu_ids)
            cudnn.benchmark = True
        else:
            net = net.to(device)

    if args.resume:
        if not args.resume_checkpoint:
            args.resume_checkpoint = args.save_dir

        if osp.isdir(args.resume_checkpoint):
            ckpt = ''
            epoch = 0
            for fn in os.listdir(args.resume_checkpoint):
                if not fn.endswith('.t7'):
                    continue

                splits = fn.rsplit('-', 2)
                t_epoch = int(splits[1])

                if t_epoch > epoch:
                    epoch = t_epoch
                    ckpt = fn

            args.resume_checkpoint = osp.join(args.resume_checkpoint, ckpt)

        if not osp.exists(args.resume_checkpoint):
            print("===> Resume checkpoint not found: ", args.resume_checkpoint)
            print("===> Exit")
            return

        # Load checkpoint.
        print('==> Resuming from checkpoint: ', args.resume_checkpoint)
        checkpoint = torch.load(args.resume_checkpoint)
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=args.mom, weight_decay=args.wd)
    if args.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_epochs, eta_min=args.lr_min)
    else:
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=step_epochs, gamma=args.lr_factor)

    if not osp.exists(args.save_dir):
        os.makedirs(args.save_dir)

    log_fn = osp.join(args.save_dir, 'train-log.txt')
    loss_fn = osp.join(args.save_dir, 'train-loss.txt')

    fp_log = open(log_fn, 'w')
    fp_log.write("===> TRAIN ARGS:\n")
    fp_log.write(str(args)+'\n')
    fp_log.write("===<\n")

    fp_loss = open(loss_fn, 'w')
    loss_log_format = '{epoch} \t {lr} \t {train_loss} \t {train_acc} \t {test_loss} \t {test_acc}'
    fp_loss.write(loss_log_format + '\n')

    # Training
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        avg_loss = 0
        acc = 0

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            avg_loss = train_loss / (batch_idx + 1)
            acc = float(correct)/total

            if args.progress_bar:
                progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (avg_loss, acc*100, correct, total))

        return avg_loss, acc

    def test(epoch):
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        avg_loss = 0
        acc = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                avg_loss = test_loss / (batch_idx + 1)
                acc = float(correct)/total

                if args.progress_bar:
                    progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                 % (avg_loss, acc*100, correct, total))

        return avg_loss, acc

    for epoch in range(start_epoch, start_epoch+args.num_epochs):
        scheduler.step()
        lr = scheduler.get_lr()
        print('\n---> lr=', lr[0])
        train_loss, train_acc = train(epoch)
        test_loss, test_acc = test(epoch)

        # '{} \t {} \t {} \t {} \t {} \t {} \n'
        msg = loss_log_format.format(
            epoch=epoch, lr=lr[0],
            train_loss=train_loss, train_acc=train_acc,
            test_loss=test_loss, test_acc=test_acc)

        print('====>\n' + loss_log_format + '\n' + msg + '\n')
        fp_loss.write(msg+'\n')

        # Save checkpoint.
        if test_acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': test_acc,
                'epoch': epoch,
            }

            time.sleep(10)
            save_name = osp.join(args.save_dir, '%s-best.t7' %
                                 (args.model_prefix))
            torch.save(state, save_name)
            best_acc = test_acc

        if test_acc >= args.save_thresh:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': test_acc,
                'epoch': epoch,
            }

            time.sleep(10)
            save_name = osp.join(args.save_dir, '%s-%04d-testacc%4.2f.t7' %
                                 (args.model_prefix, epoch, test_acc*100))
            torch.save(state, save_name)

    fp_log.close()
    fp_loss.close()


if __name__ == '__main__':
    main()

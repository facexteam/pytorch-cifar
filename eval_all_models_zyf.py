#!/usr/bin/env python
# Train CIFAR10 with PyTorch.
# based on: https://github.com/kuangliu/pytorch-cifar
# extract_features_and_evaltainer: zhaoyafei (https://github.com/walkoncross, zhaoyafei0210@gmail.com)

from __future__ import print_function
import os
import os.path as osp
import sys
import argparse

from extract_features_and_eval_zyf import extract_features_and_eval


def add_arg_parser():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--root-dir', default='./',
                        type=str, help='root dir to all sub model dirs')
    parser.add_argument('--net', default='resnet20_cifar10_nofc',
                        type=str, help='network architeture')
    parser.add_argument('--emb-size', default=32,
                        type=int, help='embedding size for some network architeture')
    parser.add_argument('--gpu-ids', default='0',
                        type=str, help='which GPUs to train on, set to "0,1,2" to use multiple GPUs')

    return parser


def eval_all_models(root_dir, net, emb=0, gpu_ids='0'):
    args_templ1 = '--cifar-dir ./data --dataset cifar10 --net ' + \
        net + ' --gpu-ids ' + gpu_ids
    args_templ2 = '--cifar-dir ./data --dataset cifar100 --net ' + \
        net + ' --gpu-ids ' + gpu_ids

    if emb > 0:
        args_templ1 += ' --emb %d' % emb
        args_templ2 += ' --emb %d' % emb

    sub_dirs = os.listdir(root_dir)
    model_dirs = []
    for fn in sub_dirs:
        fn2 = osp.join(root_dir, fn)
        print('===> check for available model file under dir:', fn2)
        if osp.isdir(fn2):
            has_model = False
            for fn22 in os.listdir(fn2):
                if fn22.endswith('.t7'):
                    has_model = True
                    model_dirs.append(fn2)
                    break

    print('\n===> sub dir with available model files: ',  model_dirs)

    for sub_dir in model_dirs:
        args_str1 = args_templ1 + ' --checkpoint ' + sub_dir + \
            '  --save-dir ' + sub_dir + '/rlt-eval-cifar10'
        args_str2 = args_templ2 + ' --checkpoint ' + sub_dir + \
            '  --save-dir ' + sub_dir + '/rlt-eval-cifar100'

        args_list1 = args_str1.split()
        args_list2 = args_str2.split()

        print('===> Start to process sud_dir: ', sub_dir)

        try:
            extract_features_and_eval(args_list1)
        except Exception as e:
            print('Failed because of exception', e)

        try:
            extract_features_and_eval(args_list2)
        except Exception as e:
            print('Failed because of exception', e)


if __name__ == '__main__':
    parser = add_arg_parser()
    args = parser.parse_args()
    print('===> input args: ', args)
    eval_all_models(args.root_dir, args.net, args.emb_size, args.gpu_ids)


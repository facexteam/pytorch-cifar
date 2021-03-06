#!/usr/bin/env python
# Train CIFAR10 with PyTorch.
# based on: https://github.com/kuangliu/pytorch-cifar
# extract_features_and_evaltainer: zhaoyafei (https://github.com/walkoncross, zhaoyafei0210@gmail.com)

from __future__ import print_function
import os
import os.path as osp
import sys
import argparse

from verification_eval.draw_pr_curve_zyf import load_roc_txt, draw_all_curves_on_analysis_figure


def add_arg_parser():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--root-dir', default='./',
                        type=str, help='root dir to all sub model dirs')
    parser.add_argument('--num-threshs', default=200,
                        type=int, help='number of threshs for ROC')
    parser.add_argument('--scales', default='1,2,4,8,16,32,64',
                        type=str, help='scales')

    return parser


def draw_all_curves_together_by_scales(root_dir, scales, num_threshs):
    fname_roc = "roc_curve_%d_threshs.txt" % num_threshs

    sub_dirs = os.listdir(root_dir)
    datasets = ['cifar10', 'cifar100']

    scales = scales.strip().split(',')

    for s in scales:
    #for s in ['1']:
        key = '-s%s-' % s
        for dataset in datasets:
            roc_data_list = []
            for fn in sub_dirs:
                sub_dir = osp.join(root_dir, fn)
                print('===> process sub dir:', sub_dir)

                if osp.isdir(sub_dir) and key in fn:
                    pos1 = fn.rfind('-s')
                    pos2 = fn.find('-', pos1+6)
                    legend = fn[pos1+1:pos2]

                    for fn22 in os.listdir(sub_dir):
                        if fn22 == 'rlt-eval-'+dataset:
                            sub_dir2 = sub_dir + '/' + fn22
                            roc_file = osp.join(sub_dir2, fname_roc)
                            tp, fn, tn, fp = load_roc_txt(roc_file)

                            roc_data = {
                                'tp': tp,
                                'fn': fn,
                                'tn': tn,
                                'fp': fp,
                                'legend': legend
                            }

                            roc_data_list.append(roc_data)

            if len(roc_data_list)>1:
                draw_all_curves_on_analysis_figure(
                    roc_data_list, '_all_s'+s+'_'+dataset, save_dir=root_dir)


if __name__ == '__main__':
    parser = add_arg_parser()
    args = parser.parse_args()
    print('===> input args: ', args)

    draw_all_curves_together_by_scales(args.root_dir, args.scales, args.num_threshs)


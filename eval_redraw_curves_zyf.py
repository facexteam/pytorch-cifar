#!/usr/bin/env python
# Train CIFAR10 with PyTorch.
# based on: https://github.com/kuangliu/pytorch-cifar
# extract_features_and_evaltainer: zhaoyafei (https://github.com/walkoncross, zhaoyafei0210@gmail.com)

from __future__ import print_function
import os
import os.path as osp
import sys
import argparse

from verification_eval.draw_pr_curve_zyf import load_roc_txt, draw_analysis_figure


def add_arg_parser():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--root-dir', default='./',
                        type=str, help='root dir to all sub model dirs')
    parser.add_argument('--num-threshs', default=200,
                        type=int, help='number of threshs for ROC')

    return parser


def redraw_curves(root_dir, num_threshs):
    fname_roc = "roc_curve_%d_threshs.txt" % num_threshs

    sub_dirs = os.listdir(root_dir)

    for fn in sub_dirs:
        sub_dir = osp.join(root_dir, fn)
        print('===> process sub dir:', sub_dir)

        if osp.isdir(sub_dir):
            for fn22 in os.listdir(sub_dir):
                if fn22.startswith('rlt-eval'):
                    sub_dir2 = sub_dir + '/' + fn22
                    roc_file = osp.join(sub_dir2, fname_roc)
                    tp, fn, tn, fp = load_roc_txt(roc_file)

                    draw_analysis_figure(tp, fn, tn, fp, save_dir=sub_dir2)


if __name__ == '__main__':
    parser = add_arg_parser()
    args = parser.parse_args()
    print('===> input args: ', args)

    redraw_curves(args.root_dir, args.num_threshs)


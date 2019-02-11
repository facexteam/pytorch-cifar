#!/usr/bin/env python
# @author: zhaoyafei
from __future__ import print_function

import os
import os.path as osp
import sys


def get_checkpoint_list(root_dir):
    print('\n===> root dir: ', root_dir)
    dir_list = os.listdir(root_dir)

    ckpt_list = []
    best_ckpt = None

    ckpt_dict = {}
    epoch_list = []

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

    # print("===> all checkpoints list: \n", ckpt_list)

    return ckpt_list, best_ckpt


def bisect_ckpt_list(ckpt_list):
    acc_pos = len('testacc')
    best_acc = 0
    best_fn = ''
    last_good_fn = ''

    delete_list = []
    # keep_list = []

    for fn in ckpt_list:
        splits = fn.rsplit('-', 2)
        acc = float(splits[-1][acc_pos:acc_pos + 5])

        if acc > 1.0 and last_good_fn is '':
            last_good_fn = fn

        if best_acc < acc:
            best_acc = acc
            if best_fn != last_good_fn:
                delete_list.append(best_fn)
            best_fn = fn
        else:
            if fn != last_good_fn:
                delete_list.append(fn)

    return delete_list, best_fn, last_good_fn


if __name__ == '__main__':
    root_dir = './'

    if len(sys.argv > 1):
        root_dir = sys.argv[1]

    ckpt_list, best_ckpt = get_checkpoint_list(root_dir)

    print("===> all checkpoints list: \n", ckpt_list)
    print('===> best_ckpt: ', best_ckpt)

    delete_list, best_fn, last_good_fn = bisect_ckpt_list(ckpt_list)
    print("===> delete_list: \n", ckpt_list)
    print('===> best_fn: ', best_fn)
    print('===> last_good_fn: ', last_good_fn)

#!/usr/bin/env python
# @author: zhaoyafei
from __future__ import print_function

import os
import os.path as osp
import sys
import shutil


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
        sep2 = '.'
        acc_part = splits[-1][acc_pos:]
        if '_' in acc_part:
            sep2 = '_'
        splits2 = acc_part.split(sep2, 1)
        acc = float(splits2[0])

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


def delete_ckpt_list(root_dir, delete_list):
    for fn in delete_list:
        os.remove(osp.join(root_dir), fn)


def clean_dir(root_dir, verbose=True, delete=False):
    ckpt_list, best_ckpt = get_checkpoint_list(root_dir)

    if verbose:
        print("\n===> all checkpoints list: \n", ckpt_list)
        print('===> best_ckpt: ', best_ckpt)

    delete_list, best_fn, last_good_fn = bisect_ckpt_list(ckpt_list)

    if verbose:
        print("===> delete_list: \n", ckpt_list)
        print('===> best_fn: ', best_fn)
        print('===> last_good_fn: ', last_good_fn)

        print('===> Is best_ckpt in delete_list:', best_ckpt in delete_list)
        print('===> Is best_fn in delete_list:', best_fn in delete_list)
        print('===> Is last_good_fn in delete_list:',
              last_good_fn in delete_list)

    if delete:
        print("===> delete un-necessary checkpoints")

        delete_ckpt_list(root_dir, delete_list)

        print("===> finish deleting")


def clean_all_sub_dir(root_dir, prefix=None):
    dir_list = os.listdir(root_dir)

    for _dir in dir_list:
        full_dir = osp.join(root_dir, _dir)

        if not osp.isdir(full_dir):
            continue

        if prefix and _dir.startswith(prefix):
            clean_dir(full_dir)


if __name__ == '__main__':
    root_dir = './'

    if len(sys.argv) > 1:
        root_dir = sys.argv[1]

    clean_dir(root_dir)

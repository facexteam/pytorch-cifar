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
        # if splits[1] == 'best':  # checkpoint with best ACC
        # checkpoint with best ACC
        if splits[1] == 'best' or splits[-1].startswith('best'):
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
    best_good_ckpt = ''
    last_good_ckpt = ''

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

        if acc > 1.0 and last_good_ckpt is '':
            last_good_ckpt = fn

        if best_acc < acc:
            best_acc = acc
            if best_good_ckpt and best_good_ckpt != last_good_ckpt:
                delete_list.append(best_good_ckpt)
            best_good_ckpt = fn
        else:
            if fn != last_good_ckpt:
                delete_list.append(fn)

    return delete_list, best_good_ckpt, last_good_ckpt


def delete_ckpt_list(root_dir, delete_list, verbose=False):
    for fn in delete_list:
        full_fn = osp.join(root_dir, fn)
        print('--> delete ', full_fn)
        os.remove(full_fn)


def mklink_to_best_ckpt(root_dir, best_good_ckpt):
    if not best_good_ckpt:
        return

    splits = best_good_ckpt.split('-')
    splits[-2] = 'best'
    best_ckpt = '-'.join(splits)

    if not osp.isfile(osp.join(root_dir, best_ckpt)):
        print('\n===> make hard link {} pointing to {}'.format(
            best_ckpt, best_good_ckpt))
        os.system('ln -P ' + osp.join(root_dir, best_good_ckpt) +
                  ' ' + osp.join(root_dir, best_ckpt))


def clean_dir(root_dir, verbose=True, delete=False):
    print("\n===> clean dir", root_dir)
    ckpt_list, best_ckpt = get_checkpoint_list(root_dir)

    if verbose:
        print("\n===> all checkpoints list: \n", ckpt_list)
        print('===> best_ckpt: ', best_ckpt)

    delete_list, best_good_ckpt, last_good_ckpt = bisect_ckpt_list(ckpt_list)

    if verbose:
        print("===> delete_list: \n", delete_list)
        print('===> best_good_ckpt: ', best_good_ckpt)
        print('===> last_good_ckpt: ', last_good_ckpt)

        print('===> Is best_ckpt in delete_list:', best_ckpt in delete_list)
        print('===> Is best_good_ckpt in delete_list:',
              best_good_ckpt in delete_list)
        print('===> Is last_good_ckpt in delete_list:',
              last_good_ckpt in delete_list)

    if delete:
        print("===> delete un-necessary checkpoints")

        delete_ckpt_list(root_dir, delete_list, verbose)

        print("===> finish deleting")

    if not best_ckpt and best_good_ckpt:
        mklink_to_best_ckpt(root_dir, best_good_ckpt)


def clean_all_sub_dir(root_dir, prefix=None, verbose=False, delete=False):
    print("\n===> clean dir", root_dir)
    dir_list = os.listdir(root_dir)

    for _dir in dir_list:
        full_dir = osp.abspath(osp.join(root_dir, _dir))
        print("\n===> process dir", full_dir)
        print("\n===> is dir", osp.isdir(full_dir))

        # if not osp.isdir(osp.abspath(full_dir)):
        if not osp.isdir(full_dir):
            continue

        if prefix:
            if _dir.startswith(prefix):
                print("\n===> clean sub dir", full_dir)
               # clean_dir(full_dir)
                clean_dir(full_dir, delete=delete, verbose=verbose)
        else:
            print("\n===> clean sub dir", full_dir)
            clean_dir(full_dir, delete=delete, verbose=verbose)


def clean_all_sub_sub_dir(root_dir, prefix=None, verbose=False, delete=False):
    print("\n===> clean dir", root_dir)
    dir_list = os.listdir(root_dir)

    for _dir in dir_list:
        full_dir = osp.abspath(osp.join(root_dir, _dir))
        print("\n===> process dir", full_dir)
        print("\n===> is dir", osp.isdir(full_dir))

        # if not osp.isdir(osp.abspath(full_dir)):
        if not osp.isdir(full_dir):
            continue

        if prefix:
            if _dir.startswith(prefix):
                print("\n===> clean sub dir", full_dir)
               # clean_dir(full_dir)
                clean_all_sub_dir(full_dir, delete=delete, verbose=verbose)
        else:
            print("\n===> clean sub dir", full_dir)
            clean_all_sub_dir(full_dir, delete=delete, verbose=verbose)


if __name__ == '__main__':
    root_dir = './'
    do_delete = False
    verbose = False
    prefix = None

    if len(sys.argv) > 1:
        root_dir = sys.argv[1]

    if len(sys.argv) > 2:
        do_delete = int(sys.argv[2])

    if len(sys.argv) > 3:
        verbose = int(sys.argv[3])

    if len(sys.argv) > 4:
        prefix = sys.argv[4]

    clean_dir(root_dir, delete=do_delete, verbose=verbose)
    clean_all_sub_dir(root_dir, delete=do_delete, verbose=verbose)
    clean_all_sub_sub_dir(root_dir, prefix=prefix,
                          delete=do_delete, verbose=verbose)

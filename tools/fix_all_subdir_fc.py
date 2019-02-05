#!/usr/bin/env python
# maintainer: zhaoyafei (https://github.com/walkoncross, zhaoyafei0210@gmail.com)

from __future__ import print_function

import os
import os.path as osp
import sys

from fix_fc_cosine_and_angles import fix_train_fc_log, fix_train_loss_log


def fix_dir(dirname, rename=False):
    print('\n===> fix dir: ', dirname)

    fn_fc_log = osp.join(dirname, 'train-last-fc.txt')
    fn_loss_log = osp.join(dirname, 'train-loss.txt')

    if osp.exists(fn_fc_log):
        fixed_dict = fix_train_fc_log(fn_fc_log, rename)
        print('---> fc log fixed')

        if osp.exists(fn_loss_log):
            fix_train_loss_log(fn_loss_log, fixed_dict, rename)
            print('---> loss log fixed')
        else:
            print('---> loss log not found')

    else:
        print('---> fc log not found')


def fix_all_subdir(root_dir, prefix=None, keywords=None, rename=False):
    dir_list = os.listdir(root_dir)
    dir_list.append('./')

    for _dir in dir_list:
        sub_dir = osp.join(root_dir, _dir)
        if prefix and not _dir.startswith(prefix):
            continue

        if isinstance(keywords, str) and keywords not in _dir:
            continue

        if isinstance(keywords, list):
            skip = False
            for kw in keywords:
                if kw not in _dir:
                    skip = True
                    break

            if skip:
                continue

        if osp.isdir(sub_dir):
            fix_dir(sub_dir, rename)


if __name__ == '__main__':
    root_dir = './'
    prefix = None
    keywords = None
    rename = True

    if len(sys.argv) > 1:
        root_dir = sys.argv[1]
    if len(sys.argv) > 2:
        prefix = sys.argv[2]
    if len(sys.argv) > 3:
        keywords = sys.argv[3]

    fix_all_subdir(root_dir, prefix, keywords, rename)

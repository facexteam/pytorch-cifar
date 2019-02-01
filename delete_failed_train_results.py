#!/usr/bin/env python
# @author: zhaoyafei
from __future__ import print_function

import os
import os.path as osp
import sys


def check(root_dir, keywords=None, prefix='checkpoints'):
    print('\n===> root dir: ', root_dir)
    dir_list = os.listdir(root_dir)
    failed_trainings = []

    if not keywords:
        print('Error: keywords is None')
        return failed_trainings

    for _dir in dir_list:
        if osp.isdir(osp.join(root_dir, _dir)) and _dir.startswith(prefix):
            skip = False

            for kw in keywords:
                if kw not in _dir:
                    # print('missed keyword: ', kw)
                    skip = True
                    break
            
            if skip:
                continue

            log_fn = osp.join(root_dir, _dir, 'train-loss.txt')
            print('log_fn: ', log_fn)

            if not osp.exists(log_fn):
                print('\tmissed train-loss.txt in ', _dir)
                failed_trainings.append(_dir)
                continue

            found = 0

            with open(log_fn, 'r') as fp_log:
                line_cnt = 0
                for line in fp_log:
                    line_cnt += 1
                    if line.startswith('199'):
                        found = 1
                        break

                print('\t%d lines parsed' % line_cnt)

                if not found:
                    print('\tmissed line 199 in ', log_fn)
                    failed_trainings.append(_dir)

                fp_log.close()

    print('\n===> check results:\n')
    print('failed_trainings: ', failed_trainings)

    return failed_trainings


def delete_sub_dirs(root_dir, sub_dirs):
    for _dir in sub_dirs:
        sub_dir = osp.join(root_dir, _dir)
        print('\n===> delete sub dir: ', sub_dir)
        cmd_str = 'rm -rf ' + sub_dir
        os.system(cmd_str)

if __name__ == '__main__':
    root_dir = './'
    prefix = 'checkpoints'
    keywords = None
    do_delete = False

    if len(sys.argv) > 1:        
        if sys.argv[1] == '-d':
            do_delete = True
            keywords = sys.argv[2:]
        else:
            keywords = sys.argv[1:]
    
    failed_trainings = check(root_dir, keywords, prefix)

    if do_delete:
        delete_sub_dirs(root_dir, failed_trainings)

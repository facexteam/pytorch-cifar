#!/usr/bin/env python
# @author: zhaoyafei
from __future__ import print_function

import os
import os.path as osp
import sys


def check(root_dir, keywords=None, prefix='train-log'):
    print('\n===> root dir: ', root_dir)
    dir_list = os.listdir(root_dir)
    failed_trainings = []
        
    for _dir in dir_list:
        log_fn = osp.join(root_dir, _dir)

        if osp.isfile(log_fn) and _dir.startswith(prefix):
            print('===> check log file: ', log_fn)
            skip = False

            if isinstance(keywords, list):
                for kw in keywords:
                    if kw not in _dir:
                        # print('missed keyword: ', kw)
                        skip = True
                        break

            elif isinstance(keywords, str):
                if keywords not in _dir:
                    skip = True
            
            if skip:
                print('...skip...')
                continue

            with open(log_fn, 'r') as fp_log:
                for line in fp_log:
                    if 'error' in line or 'Error' in line:
                        failed_trainings.append(_dir)

                        print('\terror in log file')
                        break

                fp_log.close()

    print('\n===> check results:\n')
    print('failed_trainings: ', failed_trainings)

    return failed_trainings


if __name__ == '__main__':
    root_dir = './'
    prefix = 'train-log'
    keywords = None

    if len(sys.argv) > 1:        
        keywords = sys.argv[1:]
    
    failed_trainings = check(root_dir, keywords, prefix)

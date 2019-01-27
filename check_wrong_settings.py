#!/usr/bin/env python
# @author: zhaoyafei
from __future__ import print_function

import os
import os.path as osp


def check(root_dir, prefix='checkpoints', keyword='lmcos'):

    fn_list = os.listdir(root_dir)

    for fn in fn_list:
        if osp.isdir(fn) and fn.startswith(prefix) and 'lmcos' in fn:
            print('\n===> Check sub-dir name: ', fn)
            print('---> Parse sub-dir name: ', fn)
            m_val = 0
            s_val = 0
            splits = fn.split('-')
            for spl in splits:
                spl = spl.strip()
                if spl.startswith('s'):
                    s_val = float(spl[1:])

                if spl.startswith('m'):
                    m_val = float(spl[1:])

            print('---> Parsed params from sub-dir name: s=%f, m=%f' %
                  (s_val, m_val))

            fn_log = osp.join(root_dir, fn, 'train-log.txt')
            print('---> Parse train log file: ', fn_log)
            fp_log = open(fn_log, 'r')

            m_val2 = 0
            s_val2 = 0

            for line in fp_log:
                if 'loss_m' in line:
                    splits = line.split(',')
                    for spl in splits:
                        spl = spl.strip()

                        if spl.startswith('loss_m'):
                            m_val2 = spl.rsplit('=', 1)[-1]
                            m_val2 = float(m_val2)

                        if spl.startswith('loss_scale'):
                            s_val2 = spl.rsplit('=', 1)[-1]
                            s_val2 = float(s_val2)

                    break

            print('---> Parsed params from train log: s=%f, m=%f' %
                  (s_val2, m_val2))

            if m_val != m_val2 or s_val != s_val2:
                print("wrong settings: s, m = ", s_val2, m_val2)
                print("correct settings: s, m = ", s_val, m_val)


if __name__ == "__main__":
    root_dir = './'
    prefix = 'checkpoints'
    keyword = 'lmcos'

    check(root_dir, prefix, keyword)

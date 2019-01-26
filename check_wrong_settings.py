#!/usr/bin/env python
# @author: zhaoyafei
from __future__ import print_function

import os
import os.path as osp

root_dir = './'
prefix = 'checkpoints'

fn_list = os.listdir(root_dir)

for fn in fn_list:
    if fn.startswith('checkpoints') and 'lmcos' in fn:
        print('===> Check ', fn)
        m_val = 0
        s_val = 0
        splits = fn.split('-')
        for spl in splits:
            spl = spl.strip()
            if spl.startswith('s'):
                s_val = float(spl[1:])

            if spl.startswith('m'):
                m_val = float(spl[1:])

        fn_log = osp.join(root_dir, fn, 'train-log.txt')
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

        if m_val != m_val2 or s_val != s_val2:
            print("wrong settings: s, m = ", s_val2, m_val2)
            print("correct settings: s, m = ", s_val, m_val)

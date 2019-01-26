#!/usr/bin/env python
# @author: zhaoyafei
from __future__ import print_function

import os
import os.path as osp
import numpy as np


log_fn_template = 'train-log-s%d-m'

# scale_list = [1, 2, 4, 8, 16, 32, 64]
scale_list = [64, 32, 16, 8, 4, 2, 1]
m_list = np.arange(0, 1.05, 0.05)


def check(root_dir):
    failed_m_list = []
    failed_s_list = []

    for s in scale_list:
        for m in m_list:
            print('\n===> summary train results with setting s=', s)
            if m == 0:
                log_fn = (log_fn_template + '0.txt') % (s)
            elif m == 1.0:
                log_fn = (log_fn_template + '1.0.txt') % (s)
            else:
                log_fn = (log_fn_template + '%g.txt') % (s, m)

            log_fn = osp.join(root_dir, log_fn)
            print('long_fn: ', log_fn)

            if not osp.exists(log_fn):
                failed_m_list.append(float('%g' % m))
                failed_s_list.append(s)

                print('\tmissed log file for s=%d, m=%g' % (s, m))
                continue

            with open(log_fn, 'r') as fp_log:
                for line in fp_log:
                    if 'error' in line or 'Error' in line:
                        failed_m_list.append(float('%g' % m))
                        failed_s_list.append(s)

                        print('\terror in log file for s=%d, m=%g' % (s, m))
                        break

                fp_log.close()

    print('\n===> check results:\n')
    print('failed_s_list: ', failed_s_list)
    print('failed_m_list: ', failed_m_list)

    failed_s_str = '('
    for s in failed_s_list:
        failed_s_str += str(s) + '\t'
    failed_s_str += ')'

    failed_m_str = '('
    for m in failed_m_list:
        failed_m_str += '%g\t' % m
    failed_m_str += ')'

    print('failed_s_list: ', failed_s_str)
    print('failed_m_list: ', failed_m_str)

    return failed_s_list, failed_m_list


if __name__ == '__main__':
    root_dir = './'

    check(root_dir)

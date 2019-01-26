#!/usr/bin/env python
# @author: zhaoyafei
from __future__ import print_function

import os
import os.path as osp
import numpy as np


#save_dir_template = 'checkpoints-res20-cifar-lmcos-s%d-coslr-200ep-new-m%g'
save_dir_template = 'checkpoints-res20-cifar-lmcos-s%d-coslr-200ep-new-m'

#scale_list = [1, 2, 4, 8, 16, 32, 64]
scale_list = [64, 32, 16, 8, 4, 2, 1]
m_list = np.arange(0, 1.05, 0.05)


def parse_log(log_fn, fp, s, m, write_head=1):
    print('---> Parsing log file: ', log_fn)
    line_cnt = 0
    found = 0

    with open(log_fn, 'r') as fp_log:
        if write_head:
            line = fp_log.readline()
            fp.write('Train-script\tScale\tMargin-m\t ' +
                     line.replace(' ', ''))
            write_head = 0

        for line in fp_log:
            line_cnt += 1
            if line.startswith('199'):
                fp.write('Lmcos-v1\t%d\t%g\t200\t' %
                         (s, m) + (line[3:].strip().replace(' ', ''))+'\n')
                found = 1
                break

        print('\t%d lines parsed' % line_cnt)

        if not found:
            print('\tmissed line 199 for: s=%d,m=%g' % (s, m))

        fp_log.close()

    return write_head, found


def summary_by_scales(root_dir):
    failed_s_list = []
    failed_m_list = []

    for s in scale_list:
        print('\n===> summary train results with setting s=', s)
        save_fn = 'summary_s%d.tsv.txt' % s
        write_head = 1
        with open(save_fn, 'w') as fp:
            for m in m_list:
                if m == 0:
                    _dir = (save_dir_template + '0') % (s)
                elif m == 1:
                    _dir = (save_dir_template + '1.0') % (s)
                else:
                    _dir = (save_dir_template + '%g') % (s, m)

                log_fn = osp.join(root_dir, _dir, 'train-loss.txt')
                print('long_fn: ', log_fn)

                if not osp.exists(log_fn):
                    print('\tmissed save_dir for: s=%d,m=%g' % (s, m))

                    failed_m_list.append(float('%g' % m))
                    failed_s_list.append(s)

                    continue

                write_head, found = parse_log(log_fn, fp, s, m, write_head)

                if not found:
                    failed_m_list.append(float('%g' % m))
                    failed_s_list.append(s)

            fp.close()

    print('\n===> check results:\n')
    print('failed_s_list: ', failed_s_list)
    print('failed_m_list: ', failed_m_list)

    failed_s_str = '('
    for s in failed_s_list:
        failed_s_str += str(s) + ' \t'
    failed_s_str += ')'

    failed_m_str = '('
    for m in failed_m_list:
        failed_m_str += '%g \t' % m
    failed_m_str += ')'

    print('failed_s_list: ', failed_s_str)
    print('failed_m_list: ', failed_m_str)


def summary_by_margins(root_dir):
    for m in m_list:
        print('\n===> summary train results with margins m=', m)
        save_fn = 'summary_m%g.tsv.txt' % m
        write_head = 1
        with open(save_fn, 'w') as fp:
            for s in scale_list:
                if m == 0:
                    _dir = (save_dir_template + '0') % (s)
                elif m == 1:
                    _dir = (save_dir_template + '1.0') % (s)
                else:
                    _dir = (save_dir_template + '%g') % (s, m)

                log_fn = osp.join(root_dir, _dir, 'train-loss.txt')
                print('long_fn: ', log_fn)

                if not osp.exists(log_fn):
                    print('\tmissed save_dir for: s=%d,m=%g' % (s, m))
                    continue

                write_head, found = parse_log(log_fn, fp, s, m, write_head)

            fp.close()


if __name__ == '__main__':
    root_dir = './'

    summary_by_margins(root_dir)
    summary_by_scales(root_dir)

#!/usr/bin/env python
# maintainer: zhaoyafei (https://github.com/walkoncross, zhaoyafei0210@gmail.com)

from __future__ import print_function

import os
import os.path as osp
import numpy as np
import torch


def fix_train_fc_log(fname, verbose=False):
    fn, ext = osp.splitext(fname)
    res_fn = fn + '-fixed' + ext

    fp = open(fname, 'r')
    fp_out = open(res_fn, 'w')

    line_cnt = 0

    # fc_log_format = '{epoch}\t{avg_fc_cos_max}\t{avg_fc_ang_min}\t{fc_cos_max}\t{fc_ang_min}\t{fc_cos_mat}\t{fc_ang_mat}\t{fc_wt}'
    fc_log_format = ''
    log_keys = []

    fixed_dict = {}

    for line in fp:
        line_cnt += 1
        if line_cnt == 1:
            fc_log_format = line
            splits = fc_log_format.strip().split('\t')

            for val in splits:
                log_keys.append(val[1:-1])

            fp_out.write(line)
        else:
            line_dict = {}
            line_splits = line.strip().split('\t')

            for i, key in enumerate(log_keys):
                line_dict[key] = line_splits[i]

            epoch = line_dict['epoch']

            fc_cos_mat = np.mat(
                line_dict['fc_cos_mat'], dtype=np.float32).reshape(10, -1)
            fc_cos_mat = torch.from_numpy(fc_cos_mat)
            fc_ang_mat = np.mat(
                line_dict['fc_ang_mat'], dtype=np.float32).reshape(10, -1)
            fc_ang_mat = torch.from_numpy(fc_ang_mat)

            fc_cos_mat2 = fc_cos_mat - torch.eye(fc_cos_mat.shape[0])*10
            fc_cos_max, pos = fc_cos_mat2.max(dim=0)
            fc_ang_min = fc_ang_mat[pos].diag()

            avg_fc_cos_max = fc_cos_max.mean().item()
            avg_fc_ang_min = fc_ang_min.mean().item()

            if verbose:
                print('\n===> epoch: ', epoch)
                print('---> fixed fc_cos_max: ', fc_cos_max)
                print('---> fixed fc_ang_min: ', fc_ang_min)
                print('---> fixed avg_fc_cos_max: ', avg_fc_cos_max)
                print('---> fixed avg_fc_ang_min: ', avg_fc_ang_min)

            line_dict['fc_cos_max'] = fc_cos_max.tolist()
            line_dict['fc_ang_min'] = fc_ang_min.tolist()
            line_dict['avg_fc_cos_max'] = avg_fc_cos_max
            line_dict['avg_fc_ang_min'] = avg_fc_ang_min

            fixed_dict[epoch] = {
                'avg_fc_cos_max': avg_fc_cos_max,
                'avg_fc_ang_min': avg_fc_ang_min
            }

            write_line = fc_log_format.format(**line_dict)
            fp_out.write(write_line)

    fp.close()
    fp_out.close()

    return fixed_dict


def fix_train_loss_log(fname, fixed_dict):
    fn, ext = osp.splitext(fname)
    res_fn = fn + '-fixed' + ext

    fp = open(fname, 'r')
    fp_out = open(res_fn, 'w')

    line_cnt = 0

    # fc_log_format = '{epoch}\t{avg_fc_cos_max}\t{avg_fc_ang_min}\t{fc_cos_max}\t{fc_ang_min}\t{fc_cos_mat}\t{fc_ang_mat}\t{fc_wt}'
    fc_log_format = ''
    log_keys = []

    for line in fp:
        line_cnt += 1
        if line_cnt == 1:
            fc_log_format = line
            splits = fc_log_format.strip().split('\t')

            for val in splits:
                log_keys.append(val[1:-1])

            fp_out.write(line)
        else:
            line_dict = {}
            line_splits = line.strip().split('\t')

            for i, key in enumerate(log_keys):
                line_dict[key] = line_splits[i]

            epoch = line_dict['epoch']
            line_dict['avg_fc_cos_max'] = fixed_dict[epoch]['avg_fc_cos_max']
            line_dict['avg_fc_ang_min'] = fixed_dict[epoch]['avg_fc_ang_min']

            write_line = fc_log_format.format(**line_dict)
            fp_out.write(write_line)

    fp.close()
    fp_out.close()


if __name__ == '__main__':
    fn_fc_log = './train-last-fc.txt'
    fn_loss_log = './train-loss.txt'

    fixed_dict = fix_train_fc_log(fn_fc_log)

    if osp.exists(fn_loss_log):
        fix_train_loss_log(fn_loss_log, fixed_dict)

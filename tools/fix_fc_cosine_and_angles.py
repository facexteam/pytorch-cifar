#!/usr/bin/env python
# maintainer: zhaoyafei (https://github.com/walkoncross, zhaoyafei0210@gmail.com)

from __future__ import print_function

import os
import os.path as osp
import numpy as np
import torch

import shutil


def fix_train_fc_log(fname, rename=False, verbose=False):
    fn, ext = osp.splitext(fname)

    old_fn = fname

    if rename:
        fname = fn + '-unfixed' + ext
        if osp.exists(fname):
            i = 0
            while True:
                i+=1
                fname = fn + '-unfixed' + str(i) + ext
                if not osp.exists(fname):
                    break

        shutil.move(old_fn, fname)

        print('\n===> Rename {} into {}'.format(old_fn, fname))

        fixed_fn = old_fn
    else:
        fixed_fn = fn + '-fixed' + ext
        if osp.exists(fixed_fn):
            i = 0
            while True:
                i += 1
                fixed_fn = fn + '-fixed' + str(i) + ext
                if not osp.exists(fixed_fn):
                    break

    print('\n===> Save fixed results into ', fixed_fn)

    fp = open(fname, 'r')
    fp_out = open(fixed_fn, 'w')

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


def fix_train_loss_log(fname, fixed_dict, rename=False):
    fn, ext = osp.splitext(fname)

    old_fn = fname

    if rename:
        fname = fn + '-unfixed' + ext
        if osp.exists(fname):
            i = 0
            while True:
                i += 1
                fname = fn + '-unfixed' + str(i) + ext
                if not osp.exists(fname):
                    break

        shutil.move(old_fn, fname)

        print('\n===> Rename {} into {}'.format(old_fn, fname))

        fixed_fn = old_fn
    else:
        fixed_fn = fn + '-fixed' + ext
        if osp.exists(fixed_fn):
            i = 0
            while True:
                i += 1
                fixed_fn = fn + '-fixed' + str(i) + ext
                if not osp.exists(fixed_fn):
                    break

    print('\n===> Save fixed results into ', fixed_fn)

    fp = open(fname, 'r')
    fp_out = open(fixed_fn, 'w')

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

    fixed_dict = fix_train_fc_log(fn_fc_log, True)

    if osp.exists(fn_loss_log):
        fix_train_loss_log(fn_loss_log, fixed_dict, True)

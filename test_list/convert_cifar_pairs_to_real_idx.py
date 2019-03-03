#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import os.path as osp
import sys


def load_cifar_testset(fn, n_classes=10, n_per_classes=1000):
    fp = open(fn, 'r')
    fp.readline()

    cls_indexes_mat = np.zeros((n_classes, n_per_classes), dtype=np.int32)
    index_pos_mat = np.zeros(n_classes, dtype=np.int32)

    for line in fp:
        splits = line.strip().split()
        idx = int(splits[0])
        label = int(splits[1])

        cls_indexes_mat[label][index_pos_mat[label]] = idx
        index_pos_mat[label] += 1

    print('===> index_pos_mat: \n', index_pos_mat)
    print('===> cls_indexes_mat: \n', cls_indexes_mat)

    return cls_indexes_mat


def convert_cifar_pairs(label_fn, pairs_fn):

    if 'cifar100' in label_fn:
        n_classes = 100
        n_per_classes = 100
    else:
        n_classes = 10
        n_per_classes = 1000

    cls_indexes_mat = load_cifar_testset(label_fn, n_classes, n_per_classes)

    fp = open(pairs_fn, 'r')

    prefix, ext = osp.splitext(pairs_fn)
    save_fn = prefix + '_real_idx' + ext
    fp2 = open(save_fn, 'w')

    for line in fp:
        spl = line.strip().split()

        idx1 = int(spl[0])
        idx2 = int(spl[1])
        label = int(spl[2])

        cls1 = idx1 / n_per_classes
        idx11 = idx1 % n_per_classes

        cls2 = idx2 / n_per_classes
        idx22 = idx2 % n_per_classes

        real_idx1 = cls_indexes_mat[cls1][idx11]
        real_idx2 = cls_indexes_mat[cls2][idx22]

        write_line = '%4d\t%4d\t%4d\n' % (real_idx1, real_idx2, label)

        fp2.write(write_line)

    fp.close()
    fp2.close()

    print('===> Conversion finished!')


if __name__ == '__main__':
    # label_fn = './cifar10_testset_labels.txt'
    # n_classes = 10
    # n_per_classes = 1000

    # cls_indexes_mat = load_cifar_testset(label_fn, n_classes, n_per_classes)

    label_fn = sys.argv[1]
    pairs_fn = sys.argv[2]
    convert_cifar_pairs(label_fn, pairs_fn)

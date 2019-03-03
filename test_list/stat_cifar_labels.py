#! /usr/bin/env python

from __future__ import print_function
import os.path as osp
import numpy as np


def stat_cifar_lables(cifar_dir, dataset='cifar10'):
    fn = dataset + '_testset_labels.txt'
    fp = open(fn, 'r')
    fp.readline()

    if dataset == 'cifar100':
        n_classes = 100

    else:
        n_classes = 10

    hist = np.zeros(n_classes)

    for line in fp:
        splits = line.strip().split()
        idx = int(splits[0])
        label = int(splits[1])

        hist[label] += 1

    fp.close()

    print('label hist: \n', hist)


if __name__ == '__main__':
    cifar_dir = '../data'
    dataset = 'cifar10'

    stat_cifar_lables(cifar_dir, dataset)

    dataset = 'cifar100'

    stat_cifar_lables(cifar_dir, dataset)

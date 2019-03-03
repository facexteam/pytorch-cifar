#! /usr/bin/env python

from __future__ import print_function
import os.path as osp
import torch
import torchvision


def get_cifar_lables(cifar_dir, dataset='cifar10'):
    do_download = True
    fn = dataset + '_testset_labels.txt'
    fp = open(fn, 'w')
    fp.write('idx\tlabel\n')

    if dataset == 'cifar100':
        if osp.exists(osp.join(cifar_dir, 'cifar-100-python.tar.gz')):
            print('cifar100 has already been downloaded to ', cifar_dir)
            do_download = False

        testset = torchvision.datasets.CIFAR100(
            root=cifar_dir, train=False, download=do_download, transform=None)

    else:
        if osp.exists(osp.join(cifar_dir, 'cifar-10-python.tar.gz')):
            print('cifar10 has already been downloaded to ', cifar_dir)
            do_download = False

        testset = torchvision.datasets.CIFAR10(
            root=cifar_dir, train=False, download=do_download, transform=None)

    # print('---> testset[0]: ', testset[0])

    # for i in range(10):
    for i in range(len(testset)):
        # print('---> idx %04d: label %04d ' % (i, testset[i][1]))
        fp.write('%d\t%d\n' % (i, testset[i][1]))

    fp.close()


if __name__ == '__main__':
    cifar_dir = '../data'
    dataset = 'cifar10'

    get_cifar_lables(cifar_dir, dataset)

    dataset = 'cifar100'

    get_cifar_lables(cifar_dir, dataset)

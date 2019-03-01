#!/usr/bin/env python

from __future__ import print_function
import random


rand_gen = random.Random()


# def generate_same_pairs(n_classes, n_per_classes):
#     total_pairs_list = []

#     for i in range(n_classes):
#         cnt = 0

#         pair_code_list = range(n_per_classes * (n_per_classes - 1))
#         rand_gen.shuffle(pair_code_list)

#         for pair_code in pair_code_list[0:n_per_classes-1]:
#             a = pair_code / n_per_classes
#             b = pair_code % n_per_classes

#             if a > b:
#                 a, b = b, a

#             pair = (a+i*n_per_classes, b+i*n_per_classes)
#             total_pairs_list.append(pair)
#             # print('--> pair: ', pair)
#             cnt = cnt + 1
#             # print('--> cnt: ', cnt)

#     return total_pairs_list


def generate_same_pairs(n_classes, n_per_classes):
    total_pairs_list = []

    for i in range(n_classes):
        cnt = 0

        pair_code_list = []
        pairs_list = []

        while(cnt < n_per_classes-1):
            a = rand_gen.randint(0, n_per_classes - 1)
            b = rand_gen.randint(0, n_per_classes - 2)
            if b >= a:
                b = b + 1

            if a > b:
                a, b = b, a

            pair_code = a * n_per_classes + b

            if pair_code not in pair_code_list:
                pair = (a, b)
                # print('--> pair: ', pair)
                pairs_list.append(pair)
                pair_code_list.append(pair_code)
                cnt = cnt + 1
            # print('--> cnt: ', cnt)

        for (a, b) in pairs_list:
            pair = (a+i*n_per_classes, b+i*n_per_classes)
            total_pairs_list.append(pair)

    return total_pairs_list


def generate_diff_pairs(n_classes, n_per_classes):
    total_pairs_list = []

    cnt = 0

    for i in range(n_classes):
        for j in range(i+1, n_classes):
            a = rand_gen.randint(0, n_per_classes - 1)
            b = rand_gen.randint(0, n_per_classes - 1)

            c = n_per_classes - 1 - a
            d = n_per_classes - 1 - b

            pair = (a + i*n_per_classes, b + j * n_per_classes)
            total_pairs_list.append(pair)
            cnt += 1

            pair = (c + i*n_per_classes, d + j * n_per_classes)
            total_pairs_list.append(pair)
            cnt += 1

    return total_pairs_list


if __name__ == '__main__':
    import os.path as osp

    n_classes = 100
    n_per_classes = 100
    # n_classes = 10
    # n_per_classes = 10

    same_pairs = generate_same_pairs(n_classes, n_per_classes)

    print('len(same_pairs) = ', len(same_pairs))
    # print(same_pairs)

    diff_pairs = generate_diff_pairs(n_classes, n_per_classes)

    print('len(diff_pairs) = ', len(diff_pairs))
    # print(diff_pairs)

    save_fn = './cifar100_test_pairs-same%d-diff%d.txt' % (
        len(same_pairs), len(diff_pairs))

    fp = open(save_fn, 'w')
    for pair in same_pairs:
        fp.write('%d\t%d\t%d\n' % (pair[0], pair[1], 1))

    for pair in diff_pairs:
        fp.write('%d\t%d\t%d\n' % (pair[0], pair[1], 0))

    fp.close()

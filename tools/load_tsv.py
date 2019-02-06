#!/usr/bin/env python
# maintainer: zhaoyafei (https://github.com/walkoncross, zhaoyafei0210@gmail.com)

from __future__ import print_function

import os
import os.path as osp
from collections import OrderedDict


def load_tsv(tsv_fn, with_header=True, verbose=False):
    if verbose:
        print('\n===> Load tsv file: ', tsv_fn)
        print('       with_header: ', with_header)

    fp = open(tsv_fn, 'r')

    line_cnt = 0
    field_keys = []
    parsed_lines = []

    for line in fp:
        line_cnt += 1
        if line_cnt == 1:
            splits = line.strip().split('\t')

            if with_header:
                for val in splits:
                    #field_keys.append(val[1:-1])
                    val = val.strip()
                    if val.startswith('{'):
                        val = val[1:]
                    if val.endswith('}'):
                        val = val[0:-1]

                    field_keys.append(val)

                continue
            else:
                for i in range(len(splits)):
                    key = 'f_'+str(i)  # f_0, f_1, f_2, ...
                    field_keys.append(key)

        line_dict = OrderedDict()
        line_splits = line.strip().split('\t')

        for i, key in enumerate(field_keys):
            line_dict[key] = line_splits[i]

        parsed_lines.append(line_dict)
        # write_line = header_line.format(**line_dict)
        # fp_out.write(write_line)

    fp.close()
    # fp_out.close()

    if verbose:
        print('---> %d lines parsed' % line_cnt)

    return field_keys, parsed_lines


if __name__ == '__main__':
    tsv_fn = './train-loss.txt'

    field_keys, parsed_lines = load_tsv(tsv_fn, True)
    print('\n---> field_keys: ', field_keys)
    print('\n---> parsed_lines[0]: ', parsed_lines[0])
    print('\n---> parsed_lines[-2]: ', parsed_lines[-1])

    field_keys, parsed_lines = load_tsv(tsv_fn, False)
    print('\n---> field_keys: ', field_keys)
    print('\n---> parsed_lines[0]: ', parsed_lines[0])
    print('\n---> parsed_lines[-2]: ', parsed_lines[-1])

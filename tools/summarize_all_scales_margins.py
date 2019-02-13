
#!/usr/bin/env python
# @author: zhaoyafei
from __future__ import print_function

import os
import os.path as osp
import numpy as np

from load_tsv import load_tsv


def summarize_all_scales_margins(save_prefix,
                                 scale_list,
                                 m_list):
    scale_list.sort()
    m_list.sort()

    summ_fields = [
        'avg_fc_ang_min',
        'train_ang', 'test_ang',
        'train_acc', 'test_acc',
        'avg_fc_cos_max',
        'train_cos', 'test_cos',
        'train_loss', 'test_loss',
    ]

    save_fn = save_prefix + '_all_sm.tsv.txt'
    print('\n===> summrize train results by all scales and margins')
    print('        will save results into file: ', save_fn)

    fp_out = open(save_fn, 'w')
    lines_per_subtable = 35

    main_header_row = '\t'
    for s in scale_list:
        main_header_row += '\tS_' + str(s)

    main_header_row += '\n'
    fp_out.write(main_header_row)

    sub_header_row = '\t'
    for s in scale_list:
        sub_header_row += '\t' + str(s)

    sub_header_row += '\n'

    # all_scale_dict = {}
    # for s in scale_list:
    #     tsv_fn = save_prefix + '_s%d.tsv.txt' % s
    #     print('\n---> load tsv file:', tsv_fn)

    #     field_keys, tsv_lines = load_tsv(tsv_fn)
    #     all_scale_dict[str(s)] = tsv_lines

    all_margin_dict = {}
    for m in m_list:
        tsv_fn = save_prefix + '_m%g.tsv.txt' % m
        print('\n---> load tsv file:', tsv_fn)

        field_keys, tsv_lines = load_tsv(tsv_fn)
        print('---> field_keys:', field_keys)
        print('---> tsv_lines:', tsv_lines)

        tsv_line_dict = {}
        for line in tsv_lines:
            tsv_line_dict[line['Scale']] = line

        all_margin_dict[str(m)] = tsv_line_dict

    fp_out.write('\n\navg_fc_ang_min-train_ang\n')
    fp_out.write(sub_header_row)

    for m in m_list:
        write_line = 'M_%g\t%g' % (m, m)
        if str(m) in all_margin_dict:
            for s in scale_list:
                tmp = all_margin_dict[str(m)].get(str(s), None)
                if tmp is not None:
                    write_line += '\t%f' % (float(tmp['avg_fc_ang_min']
                                                  ) - float(tmp['train_ang']))
                else:
                    write_line += '\tN.A.'

        fp_out.write(write_line+'\n')

    for i in range(lines_per_subtable - len(m_list)):
        fp_out.write('\n')

    fp_out.flush()

    fp_out.write('\n\navg_fc_ang_min-test_ang\n')
    fp_out.write(sub_header_row)

    for m in m_list:
        write_line = 'M_%g\t%g' % (m, m)
        if str(m) in all_margin_dict:
            for s in scale_list:
                tmp = all_margin_dict[str(m)].get(str(s), None)
                if tmp is not None:
                    write_line += '\t%f' % (float(tmp['avg_fc_ang_min']
                                                  ) - float(tmp['test_ang']))
                else:
                    write_line += '\tN.A.'

        fp_out.write(write_line+'\n')

    for i in range(lines_per_subtable - len(m_list)):
        fp_out.write('\n')

    for ff in summ_fields:
        fp_out.write('\n\n%s\n' % ff)
        fp_out.write(sub_header_row)

        for m in m_list:
            write_line = 'M_%g\t%g' % (m, m)
            if str(m) in all_margin_dict:
                for s in scale_list:
                    tmp = all_margin_dict[str(m)].get(str(s), None)
                    if tmp is not None:
                        write_line += '\t%s' % tmp[ff]
                    else:
                        write_line += '\tN.A.'

            fp_out.write(write_line+'\n')

        for i in range(lines_per_subtable - len(m_list)):
            fp_out.write('\n')

        fp_out.flush()

    fp_out.write('\n\ntrain_cos-avg_fc_cos_max\n')
    fp_out.write(sub_header_row)

    for m in m_list:
        write_line = 'M_%g\t%g' % (m, m)
        if str(m) in all_margin_dict:
            for s in scale_list:
                tmp = all_margin_dict[str(m)].get(str(s), None)
                if tmp is not None:
                    write_line += '\t%f' % (float(tmp['train_cos']) -
                                            float(tmp['avg_fc_cos_max']))
                else:
                    write_line += '\tN.A.'

        fp_out.write(write_line+'\n')

    for i in range(lines_per_subtable - len(m_list)):
        fp_out.write('\n')

    fp_out.flush()

    fp_out.write('\n\ntest_cos-avg_fc_cos_max\n')
    fp_out.write(sub_header_row)

    for m in m_list:
        write_line = 'M_%g\t%g' % (m, m)
        if str(m) in all_margin_dict:
            for s in scale_list:
                tmp = all_margin_dict[str(m)].get(str(s), None)
                if tmp is not None:
                    write_line += '\t%f' % (
                        float(tmp['test_cos'])- float(tmp['avg_fc_cos_max']))
                else:
                    write_line += '\tN.A.'

        fp_out.write(write_line+'\n')

    for i in range(lines_per_subtable - len(m_list)):
        fp_out.write('\n')
    fp_out.close()


if __name__ == '__main__':
    save_prefix = './summary'
    #scale_list = [1, 2, 4, 8, 16, 32, 64]
    scale_list = [64, 32, 16, 8, 4, 2, 1]
    m_list = np.arange(0, 1.05, 0.05)

    summarize_all_scales_margins(save_prefix, scale_list, m_list)

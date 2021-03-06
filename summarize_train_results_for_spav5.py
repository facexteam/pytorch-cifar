#!/usr/bin/env python
# @author: zhaoyafei
from __future__ import print_function

import os
import os.path as osp
import numpy as np

from summarize_train_results import summary_by_margins, summary_by_scales

if __name__ == '__main__':
    root_dir = './'
    sub_dir_template = 'checkpoints-res20-cifar-coslr-200ep-spav5-s%d-m%g-n0-b0'

    #scale_list = [1, 2, 4, 8, 16, 32, 64]
    scale_list = [64, 32, 16, 8, 4, 2, 1]
    # m_list = np.arange(0, 1.05, 0.05)
    m_list = [5, 7.5, 10, 12.5, 15]
    prefix = 'spav5_summary'

    summary_by_margins(root_dir, sub_dir_template,
                       scale_list, m_list,
                       prefix)
    summary_by_scales(root_dir, sub_dir_template,
                      scale_list, m_list,
                      prefix)


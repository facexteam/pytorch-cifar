#!/usr/bin/env python
# @author: zhaoyafei
from __future__ import print_function

import os
import os.path as osp
import numpy as np

from summarize_train_results import summary_by_margins, summary_by_scales

if __name__ == '__main__':
    root_dir = './'
    sub_dir_template = 'checkpoints-res20-cifar-coslr-200e-spa-s%d-m%g-b0'

    #scale_list = [1, 2, 4, 8, 16, 32, 64]
    scale_list = [64, 32, 16, 8, 4, 2, 1]
    # m_list = np.arange(0, 1.05, 0.05)
    m_list = [1.1, 1.2, 1.25, 1.5, 1.75, 1.8, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 4.0, 4.5, 5.0]
    prefix = 'spa_summary'

    summary_by_margins(root_dir, sub_dir_template,
                       scale_list, m_list,
                       prefix)
    summary_by_scales(root_dir, sub_dir_template,
                      scale_list, m_list,
                      prefix)

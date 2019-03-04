from draw_pr_curve_zyf import calc_roc, calc_presicion_recall, draw_analysis_figure

import numpy as np
import sys
import os
import os.path as osp


from numpy.linalg import norm


def calc_similarity(all_ftr_mat, pairs_idx_list, distance_type='cosine'):
    print "Calc similarities of pairs"

    sim_list = []
    if distance_type == 'squared':
        for idx_pair in pairs_idx_list:
            dist_vec = all_ftr_mat[idx_pair[0]] - all_ftr_mat[idx_pair[1]]
            sim = -np.dot(dist_vec, dist_vec.T)
            sim_list.append(sim)
    else:
        for idx_pair in pairs_idx_list:
            sim = np.dot(all_ftr_mat[idx_pair[0]], all_ftr_mat[idx_pair[1]])
            # print('===> In calc_similarity:')
            # print('all_ftr_mat[%d]:\n' % idx_pair[0], all_ftr_mat[idx_pair[0]])
            # print('all_ftr_mat[%d]:\n' % idx_pair[1], all_ftr_mat[idx_pair[1]])
            # print('norm of all_ftr_mat[%d]:\n' % idx_pair[0], norm(
            #     all_ftr_mat[idx_pair[0]]))
            # print('norm of all_ftr_mat[%d]:\n' % idx_pair[1], norm(
            #     all_ftr_mat[idx_pair[1]]))
            # print('sim=', sim)

            sim_list.append(sim)

    return sim_list


def load_image_pairs(pairs_fn):
    cnt = 0
    same_pairs_idx_list = []
    diff_pairs_idx_list = []

    idx_lists = [diff_pairs_idx_list, same_pairs_idx_list]

    fp = open(pairs_fn, 'r')
    for line in fp:
        splits = line.strip().split()
        idx1 = int(splits[0])
        idx2 = int(splits[1])
        label = int(splits[2])

        idx_lists[label].append((idx1, idx2))
        cnt += 1

#         print(same_pairs_idx_list)
#         print(diff_pairs_idx_list)
#         if cnt == 10:
#             break

    fp.close()
    return same_pairs_idx_list, diff_pairs_idx_list


def eval_roc_and_pr(ftr_mat, pairs_file, save_dir):
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    # do_norm = not(no_normalize)
    # ftr_mat = load_mat_features(feature_mat_file, do_norm)

    same_pairs_idx_list, diff_pairs_idx_list = load_image_pairs(
        pairs_file)

    same_sim_list = calc_similarity(ftr_mat, same_pairs_idx_list)

    fn_same = osp.join(save_dir, 'same_pairs_similarity.txt')
    fp_same = open(fn_same, 'w')

    for (i, sim) in enumerate(same_sim_list):
        fp_same.write("%s %s %g\n" %
                      (same_pairs_idx_list[i][0], same_pairs_idx_list[i][1], sim))
    fp_same.close()

    diff_sim_list = calc_similarity(ftr_mat, diff_pairs_idx_list)

    fn_diff = osp.join(save_dir, 'diff_pairs_similarity.txt')
    fp_diff = open(fn_diff, 'w')

    for (i, sim) in enumerate(diff_sim_list):
        fp_diff.write("%s %s %g\n" %
                      (diff_pairs_idx_list[i][0], diff_pairs_idx_list[i][1], sim))
    fp_diff.close()

    threshs = None
    tp, fn, tn, fp = calc_roc(same_sim_list, diff_sim_list, threshs, save_dir)
    calc_presicion_recall(tp, fn, tn, fp, threshs, save_dir)
    draw_analysis_figure(tp, fn, tn, fp, save_dir, True)

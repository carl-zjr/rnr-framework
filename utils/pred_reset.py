# -*- coding:utf-8 -*-

###########################################################################
#                                                                         #
# Program :  reset values of predicted matrix based on rnr curve          #
#                                                                         #
###########################################################################

import numpy as np
import copy
import utils.rnr as rnrer

def recalculate_pred(sec, rnr, pred_matrix, bin_num=50):
    value = []
    for i in sec:
        value.append(rnr[i])
    value = np.array(value)

    # rnr value normalization
    max_value, min_value = np.max(value), np.min(value)
    # value = (value - min_value) / (max_value - min_value) * 5
    value = (value - min_value) / (max_value - min_value)
    rnr = dict(zip(sec, value))
    rnrer.write_statistics_dict(sec, rnr, 'norm_rnr', 'section', 'value')
    
    row, column = pred_matrix.shape
    max_pred, min_pred = np.max(pred_matrix), np.min(pred_matrix)
    grap = 1 / bin_num
    
    index_matrix = np.floor((pred_matrix - min_pred) / (max_pred - min_pred) / grap).astype(np.int32)
    ret_mat = copy.deepcopy(pred_matrix)
    for i in range(row):
        for j in range(column):
            if index_matrix[i, j] == bin_num:
                ret_mat[i, j] = value[bin_num-1]
            else:
                ret_mat[i, j] = value[index_matrix[i, j]]
    return ret_mat

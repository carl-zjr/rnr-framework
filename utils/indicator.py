# -*- coding:utf-8 -*-

###########################################################################
#                                                                         #
# Program :  realize the calculation of various performance indicators    #
#                                                                         #
###########################################################################

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

def rmse_mae(pred, actual):
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return np.sqrt(mean_squared_error(pred, actual)), mean_absolute_error(pred, actual)

def recall_precision(pred, test, topN = 20):
    row, column = test.shape
    hit = 0
    recall = 0
    precision = 0
    for user in range(row):
        actual = np.argsort(-test[user, test[user].nonzero()]).flatten()
        predict = np.argsort(-pred[user, pred[user].nonzero()]).flatten()
        hit += len(set(actual).intersection(set(predict[:topN])))
        recall += actual.size 
        precision += topN
    return hit, hit/recall, hit/precision

def recall_precision_arhr(pred, test, topN = 20):
    row, column = test.shape
    hit, recall, precision, arhr = 0, 0, 0, 0
    for user in range(row):
        actual = np.argsort(-test[user, test[user].nonzero()]).flatten()
        predict = np.argsort(-pred[user, pred[user].nonzero()]).flatten()
        hit += len(set(actual).intersection(set(predict[:topN])))
        topn_list = predict[:topN].tolist()
        hit_list = list(set(actual).intersection(set(predict[:topN])))
        for h in hit_list:
            arhr += 1 / (topn_list.index(h) + 1)  
        recall += actual.size 
        precision += topN
    return hit, hit/recall, hit/precision, arhr/row

'''
def calculate_recall_precision_for_qrlatentfactor(pred_matrix, test, topN = 20):
    row, column = test.shape
    hit = 0
    recall = 0
    precision = 0
    for user in range(10):
        actual = np.argsort(-test[user, test[user].nonzero()]).flatten()
        predict = np.argsort(-pred_matrix[user, pred_matrix[user].nonzero()]).getA().flatten()
        hit += len(set(actual).intersection(set(predict[:topN])))
        recall += actual.size 
        precision += topN
    return hit, hit/recall, hit/precision

def eval_recall_precision_ARHR_for_qrlatentfactor(pred_matrix, test, topN = 20):
    row, column = test.shape
    hit, recall, precision, arhr = 0, 0, 0, 0
    for user in range(row):
        actual = np.argsort(-test[user, test[user].nonzero()]).flatten()
        predict = np.argsort(-pred_matrix[user, pred_matrix[user].nonzero()]).getA().flatten()
        hit += len(set(actual).intersection(set(predict[:topN])))
        topn_list = predict[:topN].tolist()
        hit_list = list(set(actual).intersection(set(predict[:topN])))
        for h in hit_list:
            arhr += 1 / (topn_list.index(h) + 1)  
        recall += actual.size 
        precision += topN
    return hit, hit/recall, hit/precision, arhr/row
'''

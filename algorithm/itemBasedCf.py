# -*- coding:utf-8 -*-

###########################################################################
#                                                                         #
#       Program : recommendation algorithm - item-based cf                #
#                                                                         #
###########################################################################

import numpy as np
import os

def similarity(ratings, epsilon=1e-9):
    # epsilon is a small number for handling dived-by-zero errors
    sim = ratings.T.dot(ratings) + epsilon
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)

def predict_topk_nobias(ratings, similarity, k=40):
    pred = np.zeros(ratings.shape)
    row, column = ratings.shape
    for j in range(column):
        top_k_items = np.array([np.argsort(similarity[:,j])[:-k-1:-1]])
        for i in range(row):
            pred[i, j] = similarity[j, :][top_k_items].dot(ratings[i, :][top_k_items].T) 
            pred[i, j] /= np.sum(np.abs(similarity[j, :][top_k_items]))
    return pred

def pred_matrix_writer(pred, savename):
    try:
        os.makedirs('output')
    except OSError:
        pass
    row, column = pred.shape
    with open(os.getcwd() + '/output/' + savename + '.txt', 'w') as file:
        for i in range(row):
            for j in range(column):
                file.write(str(pred[i, j]) + '\t')
            file.write('\n')

def pred_matrix_normalization(pred):
    max_pred, min_pred = np.max(pred), np.min(pred)
    pred = (pred - min_pred) / (max_pred - min_pred)
    return pred










# -*- coding:utf-8 -*-

###########################################################################
#                                                                         #
#       Main Function : implement framework for recommendation system     #
#                                                                         #
###########################################################################

import os
from utils import preprocess as ppr
from utils import matrix as mx
from algorithm import itemBasedCf as icf
from utils import indicator as cator
from utils import rnr
from utils import pred_reset as reset

def clean():
    paras_grid = [['./dataset/ml-100k/u.data', '\t', 'movielens', 5, 1, 'time', 0.1],
                  ['./dataset/opsahl-ucforum/out.opsahl-ucforum', ' ', 'ucforum', 5, 1, 'random', 0.2],
                  ['./dataset/delicious-2k/user_taggedbookmarks-timestamps.dat', '\t', 'delicious', 5, 2, 'random', 0.2],
                  ['./dataset/edit-enwikibooks/out.edit-enwikibooks', ' ', 'wiki', 5, 1, 'random', 0.2],
                  ['./dataset/lastfm-2k/user_taggedartists-timestamps.dat', '\t', 'lastfm', 2, 2, 'random', 0.2]]

    shape = ppr.handler(paras_grid[:1])
    return shape

if __name__ == '__main__':
    shape = clean()
    users, items = shape[0]
    
    filepath = './input/train_movielens'
    train = mx.rating_matrix(filepath, users, items)

    filepath = './input/test_movielens'
    test = mx.rating_matrix(filepath, users, items)

    item_similarity = icf.similarity(train)

    print('{:^10}{:^15}{:^25}{:^25}{:^25}'.format('topN', 'hit', 'recall', 'precision', 'arhr'))
    for topN in [5, 10, 20, 30, 40, 80]:
        # original predicted matrix producted by original algorithm
        pred = icf.predict_topk_nobias(train, item_similarity, k = topN)
        icf.pred_matrix_writer(pred, 'original_predicted_matrix')
        h, r, p, arhr = cator.recall_precision_arhr(pred, test, topN = topN)
        print('{:^10}{:^15}{:^25}{:^25}{:^25}'.format(topN, h, r, p, arhr))

        # pnr predicted matrix producted by pnr framework
        sec, rnr_value = rnr.rnr_calculation(train, pred, bin_num = 50)
        rnr_matrix = reset.recalculate_pred(sec, rnr_value, pred, bin_num = 50)
        icf.pred_matrix_writer(rnr_matrix, 'rnr_predicted_matrix')
        rnr_hit, rnr_recall, rnr_precision, rnr_arhr = cator.recall_precision_arhr(rnr_matrix, test, topN = topN)
        print('{:^10}{:^15}{:^25}{:^25}{:^25}'.format(topN, rnr_hit, rnr_recall, rnr_precision, rnr_arhr))

        

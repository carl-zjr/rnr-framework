# -*- coding:utf-8 -*-

########################################################################################
#                                                                                      #
# Program : calculating rnr curve based on predicted matrix created by algorithm       #
#                                                                                      #
########################################################################################

import os
import numpy as np
import pandas as pd
from itertools import groupby

def write_statistics_dict(order, dic, name, column1, column2):
    try:
        os.makedirs('output')
    except OSError:
        pass
    c1 = []
    c2 = []        
    for i in order:
        c1.append(i)
        c2.append(dic[i])
    cols = [column1, column2]
    dataframe = pd.DataFrame({column1:c1, column2:c2})
    dataframe.to_csv(os.getcwd() + '/output/' + name + ".csv", index = False, sep=',', columns = cols)

def rnr_calculation(rating, pred, bin_num):   
    row, column = rating.shape
    ex, nex = [], []
    for i in range(row):
        for j in range(column):
            if rating[i, j] != 0:
                ex.append(pred[i, j])
            else:
                nex.append(pred[i, j])
    ex = np.array(ex)
    nex = np.array(nex)
    max_ex, min_ex = np.max(ex), np.min(ex)
    max_nex, min_nex = np.max(nex), np.min(nex)
    ex = (ex - min_ex) / (max_ex - min_ex + 1e-9)
    nex = (nex - min_nex) / (max_nex - min_nex + 1e-9)

    # print('# writting existing edges statistics dictionary ...')
    sec1, ret1 = statistics(ex, bin_num, 'existing_edges')
    # print('# writting non-existing edges statistics dictionary ...')
    sec2, ret2 = statistics(nex, bin_num, 'non_existing_edges')
    # print('# writting pnr statistics dictionary ...')
    pnr = []
    for i in sec1:
        pnr.append(ret1[i] / (ret2[i] + 1e-9))
        
    write_statistics_dict(sec1, dict(zip(sec1, pnr)), 'rnr_curve', 'section', 'value')
    return sec1, dict(zip(sec1, pnr))

def statistics(score, bin_num, savename):
    gap = 1.0 / bin_num
    section_num = dict()
    for k, g in groupby(sorted(score), key=lambda x: x//gap):
        section_num.update({'{}-{}'.format(abs(k*gap), abs((k+1)*gap)):len(list(g))})
        
    for k in range(bin_num):
        if '{}-{}'.format(abs(k*gap), abs((k+1)*gap)) not in section_num.keys():
            section_num.update({'{}-{}'.format(abs(k*gap), abs((k+1)*gap)):0})

    new_section = sorted(sorted(section_num.keys(), key = lambda x:x[1], reverse = False))
    distribution = []
        
    for i in new_section:
        distribution.append(section_num[i])
    dis_ret = dict(zip(new_section, distribution))
    write_statistics_dict(new_section, dis_ret, savename, 'section', 'value')
    return new_section, dis_ret

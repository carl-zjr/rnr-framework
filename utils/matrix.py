# -*- coding:utf-8 -*-

###########################################################################
#                                                                         #
#       Program : loading data from files and creating rating martix      #
#                                                                         #
###########################################################################

import numpy as np
import pandas as pd
from utils.preprocess import data_loader
    
def rating_matrix(filepath, users, items):
    data = data_loader(filepath)
    matrix = np.zeros((users, items))
    df = pd.DataFrame(data)
    for line in df.itertuples():
        matrix[line[1]-1, line[2]-1] = 1 # line[3]
    return matrix


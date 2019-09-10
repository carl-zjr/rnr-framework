# -*- coding:utf-8 -*-

###########################################################################
#                                                                         #
#              Program : loading data and pre-process                     #
#                                                                         #
###########################################################################

import os
import random
import numpy as np

# reading data-file and return a list
def data_loader(filepath, sep='\t'):
    data = list()
    with open(filepath, 'r') as file:
        line = file.readlines()
        for i in line:
            data.append([int(j) for j in i.split(sep)])
    return data 

# writing data which has been pre-processed
# data-format : user-item-rating
def data_writer(data, savename):
    try:
        os.makedirs('input')
    except OSError:
        pass
    with open(os.getcwd() + '/input/' + savename, 'w') as file:
        for u, i, p, t in data:
            file.write(str(u) + '\t' + str(i) + '\t' + str(p) + '\n')

def display_info(dname, users, items, max_user, max_item, edges, uniedges):
    column = 40
    print('#' * column)
    print('#' + ' ' * (column-2) + '#')

    string = '# data name : {}'.format(dname)
    print(string + ' ' * (column-len(string)-1) + '#')
    string = '# users : {}'.format(users)
    print(string + ' ' * (column-len(string)-1) + '#')
    string = '# items : {}'.format(items)
    print(string + ' ' * (column-len(string)-1) + '#')
    string = '# max_user : {}'.format(max_user)
    print(string + ' ' * (column-len(string)-1) + '#')
    string = '# max_item : {}'.format(max_item)
    print(string + ' ' * (column-len(string)-1) + '#')
    string = '# edges : {}'.format(edges)
    print(string + ' ' * (column-len(string)-1) + '#')
    string = '# uniedges : {}'.format(uniedges)
    print(string + ' ' * (column-len(string)-1) + '#')
    
    print('#' + ' ' * (column-2) + '#')
    print('#' * column + '\n')

def preprocess(data, savename, user_threshold, item_threshold, division='time', test_size=0.1):
    user_item, item_user = dict(), dict()
    # create dictionarys : user-items and item-users
    # user-items data format like {345 : [1, 4, 89]}
    # item-users data format like {56 : [6, 7]}
    for u, i, r, t in data:
        if u not in user_item.keys():
            user_item.setdefault(u, [i])
        else:
            user_item[u].append(i)
        if i not in item_user.keys():
            item_user.setdefault(i, [u])
        else:
            item_user[i].append(u)
    # remove the same edges between users-set and items-set, edges like (34, 67)
    unique = dict()
    for u, i, r, t in data:
        if (u, i) not in unique.keys():
            # only care about whether there is a preference, do not care about the strength of the preference
            # so set the r = 1
            unique.setdefault((u, i), [1, t])

    user, item = list(), list()
    for u in user_item.keys():
        if len(set(user_item[u])) > user_threshold:
            user.append(u)
    for u in item_user.keys():
        if len(set(item_user[u])) > item_threshold:
            item.append(u)
    # remove sample in data but not in user-dict or item-dict above
    output = []
    for k, v in unique.items():
        u, i = k
        r, t = v
        if u in user and i in item:
            output.append([u, i, r, t])

    display_info(savename, len(user_item.keys()), len(item_user.keys()), max(user_item.keys()), max(item_user.keys()), len(data), len(output))

    if division == 'time':
        # sorted by time stamp
        temp = np.array(output)
        index = np.argsort(temp[:,3])
        output = temp[index].tolist()
    elif division == 'random':
        random.shuffle(output)
    else:
        print('[error] data division error! please check you paras.')
    
    data_writer(output, savename)
    
    # split data
    cut_point = int(len(output) * (1 - test_size))
    data_writer(output[:cut_point], 'train_' + savename)
    data_writer(output[cut_point:], 'test_' + savename)

    assert len(output[:cut_point]) + len(output[cut_point:]) == len(output)
    return max(user_item.keys()), max(item_user.keys())

def handler(paras_grid):
    shape = list()
    for i in range(len(paras_grid)):
        dset = data_loader(paras_grid[i][0], paras_grid[i][1])
        max_user, max_item = preprocess(dset, paras_grid[i][2], paras_grid[i][3], paras_grid[i][4], paras_grid[i][5], paras_grid[i][6])
        shape.append([max_user, max_item])
    return shape

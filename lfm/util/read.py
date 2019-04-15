# -*-coding:utf8-*-
"""
author: Yu
date: 20190412
util function
"""

import os
import codecs
import random


def get_item_info(input_file):
    """
    get item info: [title, genre]
    Args:
        input_file: item info file
    Return:
        a dict, key:itemid, value:[title, genre]
    """

    item = {}
    if not os.path.exists(input_file):
        return item

    with codecs.open(input_file, 'r', encoding='utf8') as fb:
        for line in fb:
            item_info = line.strip().split('::')
            if len(item_info) == 3:
                itemid, title, genre = item_info[0], item_info[1], item_info[2]
                item[itemid] = [title, genre]
            else:
                print('wrong item info: ', line)

    return item


def get_ave_score(input_file):
    """
    get item ave rating score
    Args:
        input_file: user rating file
    Return:
        item_score_dict, a dict, key: itemid, value: 所有用户对该物品评分的平均
        user_score_dict, a dict, key: userid, value: 该用于所有物品评分的平均
    """
    item_score_dict = {}
    user_score_dict = {}

    if not os.path.exists(input_file):
        return item_score_dict, user_score_dict

    item_record_dict = {}
    user_record_dict = {}

    with codecs.open(input_file, 'r', encoding='utf8') as fb:
        for line in fb:
            item = line.strip().split('::')
            if len(item) == 4:
                userid, itemid, rating = item[0], item[1], float(item[2])

                if itemid not in item_record_dict:
                    item_record_dict[itemid] = [0, 0.0]
                item_record_dict[itemid][0] += 1
                item_record_dict[itemid][1] += rating

                if userid not in user_record_dict:
                    user_record_dict[userid] = [0, 0.0]
                user_record_dict[userid][0] += 1
                user_record_dict[userid][1] += rating
            else:
                print('wrong item info: ', line)

    for itemid in item_record_dict:
        item_score_dict[itemid] = round(item_record_dict[itemid][1] / item_record_dict[itemid][0], 3)

    for userid in user_record_dict:
        user_score_dict[userid] = round(user_record_dict[userid][1] / user_record_dict[userid][0], 3)

    return item_score_dict, user_score_dict


def get_train_data(input_file):
    """
    get train data for LFM model train
    Args:
        input_file: user item rating file
    Return:
        a list, [(userid, itemid, label), (userid1, itemid1, label)]
    """
    train_data = []

    if not os.path.exists(input_file):
        return train_data

    item_score_dict, user_score_dict = get_ave_score(input_file)

    neg_dict = {}
    pos_dict = {}

    with codecs.open(input_file, 'r', encoding='utf8') as fb:
        for line in fb:
            item = line.strip().split('::')
            if len(item) == 4:
                userid, itemid, rating = item[0], item[1], float(item[2])

                if userid not in pos_dict:
                    pos_dict[userid] = []
                if userid not in neg_dict:
                    neg_dict[userid] = []

                # 根据评分决定样本是正样本还是负样本，这里引入了用户的评分基线（有的用户严格，有的用户宽松）
                if rating >= user_score_dict[userid]:
                    pos_dict[userid].append((itemid, 1))
                else:
                    neg_dict[userid].append((itemid, 0))
            else:
                print('wrong item info: ', line)

    for userid in pos_dict:
        # 为了平衡正负样本的数量，这里要取出正负样本中，样本比较少的样本数量，根据这个数量对样本比较多得做截断处理
        data_num = min(len(pos_dict[userid]), len(neg_dict.get(userid, [])))
        if data_num > 0:
            pos_samples = [(userid, info[0], info[1]) for info in pos_dict[userid]]
            random.shuffle(pos_samples)
            train_data += pos_samples[:data_num]

            neg_smaples = [(userid, info[0], info[1]) for info in neg_dict[userid]]
            random.shuffle(neg_smaples)
            train_data += neg_smaples[:data_num]
        else:
            continue

    return train_data


if __name__ == '__main__':
    train_data = get_train_data('../data/ratings.dat')
    print(len(train_data))
    for data in train_data[:80]:
        print(data)

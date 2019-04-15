# -*-coding:utf8-*-
"""
author: Yu
date: 20190415
数据相关处理
"""
from __future__ import division
import os


def get_item_cate(input_file, ave_score):
    """
    获取每个item的分类信息，从item的角度看，每个item的所属类别及对应的比例；从类别的角度看，每个类别下item平均打分的倒排序
    Args:
        input_file: item info文件
        ave_score: 每个item的平均打分的dict, key itemid, value ave_score
    Return:
        item_cate: a dict, key: itemid, value: a dict, key cate, value ratio
        cate_item: a dict, key: cate, value: items sorted by ave_score
    """
    if not os.path.exists(input_file):
        return {}, {}

    # 得到每个item对应的cate信息
    # key: itemid, value: {cate1: ratio1, cate2: ratio2}
    item_cate = {}

    with open(input_file, encoding='utf8') as fb:
        for line in fb:
            info = line.strip().split("::")
            if len(info) != 3:
                print("wrong data: ", line)
                continue

            itemid, cate_list = info[0], info[-1].strip().split("|")

            if itemid not in item_cate:
                item_cate[itemid] = {}

            ratio = round(1.0 / len(cate_list), 3)

            for cate in cate_list:
                item_cate[itemid][cate] = ratio

    # 根据cate将item归类，为的是得出每个cate下值得推荐的item排序（根据item的ave_score）
    # key: cate, value: [(itemid, ave_score)]
    cate_item = {}

    for itemid in item_cate:
        for cate in item_cate[itemid].keys():
            if cate not in cate_item:
                cate_item[cate] = []

            #
            cate_item[cate].append((itemid, ave_score.get(itemid, 0)))

    # 排序处理
    topK = 100

    for cate in cate_item:
        cate_item[cate] = sorted(cate_item[cate], key=lambda record: record[1], reverse=True)[:topK]

    return item_cate, cate_item


def get_ave_score(input_file):
    """
    计算每个item的平均打分
    Args:
        input_file: user item rating文件
    Return:
        dict, key: itemid, value: ave score
    """
    if not os.path.exists(input_file):
        return {}

    tmp_dict = {}

    with open(input_file) as fb:
        for line in fb:
            record = line.strip().split("::")

            if len(record) != 4:
                print("wrong data: ", line)
                continue

            userid, itemid, rating = record[0], record[1], float(record[2])

            if itemid not in tmp_dict:
                tmp_dict[itemid] = {"count": 0, "total_score": 0.0}

            tmp_dict[itemid]["count"] += 1
            tmp_dict[itemid]["total_score"] += rating

    ave_score_dict = {}

    for itemid in tmp_dict:
        ave_score_dict[itemid] = round(tmp_dict[itemid]["total_score"] / tmp_dict[itemid]["count"], 4)

    return ave_score_dict


if __name__ == '__main__':
    ave_score = get_ave_score("../data/ratings.dat")
    item_cate, cate_item = get_item_cate("../data/movies.dat", ave_score)

    # Children'类别下的推荐item
    print(cate_item["Children's"])

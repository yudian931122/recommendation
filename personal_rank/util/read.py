# -*-coding:utf8-*-
"""
author: Yu
date: 20190413
数据处理
"""

import os


def get_graph_from_data(input_file):
    """
    根据输入数据构造一个二分图
    Args:
        input_file: user item rating 文件
    Return:
        a dict, (UserA:{item1:1, item2:1}, item1:{UserA:1})
    """
    if not os.path.exists(input_file):
        return {}

    score_thr = 4
    graph = {}

    with open(input_file) as fb:
        for line in fb:
            record = line.strip().split("::")
            if len(record) != 4:
                print("wrong data: ", line)
                continue

            userid, itemid, rating = record[0], "item_%s" % record[1], float(record[2])

            if rating < score_thr:
                continue

            if userid not in graph:
                graph[userid] = {}

            graph[userid][itemid] = 1

            if itemid not in graph:
                graph[itemid] = {}

            graph[itemid][userid] = 1

    return graph


def get_item_info(input_file):
    """
    根据输入数据结构化item的信息
    Args:
        input_file: item信息文件
    Return:
        a dict, {itemid: (title, genre)}
    """
    if not os.path.exists(input_file):
        return {}

    item_info = {}

    with open(input_file, encoding='utf8') as fb:
        for line in fb:
            record = line.strip().split("::")

            if len(record) != 3:
                print("wrong data: ", line)
                continue

            itemid, titel, genre = record[0], record[1], record[2]

            item_info[itemid] = (titel, genre)

    return item_info


if __name__ == '__main__':
    # data_path = '../data/ratings.dat'
    # graph = get_graph_from_data(data_path)

    # print(graph['item_3409'])

    item_info = get_item_info("../data/movies.dat")
    print(len(item_info))
    print(item_info)

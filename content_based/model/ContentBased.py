# -*-coding:utf8-*-
"""
author: Yu
date: 20190415
基于内容的推荐
"""
from __future__ import division
import os
from content_based.util import read


def get_user_profile(item_cate, input_file):
    """
    计算用户对item类别的偏好
    Args:
        item_cate: 每个item的类别信息
        input_file: user item rating文件
    Return:
        user_profile: a dict, key: userid, value: list[(cate1, score1), (cate2, score2)]
    """
    if not os.path.exists(input_file):
        return {}

    score_thr = 4

    tmp_user_profile = {}

    with open(input_file) as fb:
        for line in fb:
            record = line.strip().split("::")
            if len(record) != 4:
                print("wrong data: ", line)
                continue

            userid, itemid, rating, timestamp = record[0], record[1], float(record[2]), int(record[3])

            # 只有打分大于4分的，才认为user对item是有偏好的，更常见的做法是，取user的基线（user对打分item的严格程度）作为阈值
            if rating < score_thr:
                continue

            if itemid not in item_cate:
                continue

            time_score = get_time_score(timestamp)

            if userid not in tmp_user_profile:
                tmp_user_profile[userid] = {}

            for cate in item_cate[itemid]:
                if cate not in tmp_user_profile[userid]:
                    tmp_user_profile[userid][cate] = 0
                # user对某个cate的偏好值计算
                # user_cate_profile_score = E item_rating * time_score * item_cate_ratio
                # E:对所有user有偏好的item的cate的累加
                # item_rating:user对item的打分
                # time_score:考虑了user的时间衰减，时间越近time_score分值越高
                # item_cate_ratio:item属于cate的成分是多少，按照平均分配，比如item1的cate标签为A|B|C，那么item1属于A的成分是1/3
                tmp_user_profile[userid][cate] += rating * time_score * item_cate[itemid][cate]

    # 获取最终的user profile
    user_profile = {}
    topK = 3
    for userid in tmp_user_profile:
        if userid not in user_profile:
            user_profile[userid] = []

        total_score = 0
        # 取每个user对cate偏好的前topK个cate
        for record in sorted(tmp_user_profile[userid].items(), key=lambda cate_info: cate_info[1], reverse=True)[:topK]:
            user_profile[userid].append([record[0], record[1]])
            total_score += record[1]

        # 将user对这topK个cate的偏好进行归一化处理
        for index in range(len(user_profile[userid])):
            user_profile[userid][index][1] = round(user_profile[userid][index][1] / total_score, 3)

    return user_profile


def get_time_score(timestamp):
    """
    根据时间衰减来计算时间得分
    Args:
        timestamp: 时间戳
    Return:
        score, a num
    """
    fix_timestamp = 1046454590
    total_sec = 24 * 60 * 60
    # 天数
    delta = (fix_timestamp - timestamp) / total_sec / 100
    return round(1 / (delta + 1), 5)


def recom(item_cate, user_profile, userid, topK):
    """
    给user进行推荐
    Args:
        item_cate: 每个cate下的推荐item列表
        user_profile: user对cate的偏好
        userid: 需要推荐的user
        topK: 推荐数量
    Return:
        recom_result: a dict, key: userid, value: item list
    """
    if userid not in user_profile:
        return {}

    recom_result = {}

    if userid not in recom_result:
        recom_result[userid] = []

    for record in user_profile[userid]:
        cate = record[0]
        ratio = record[1]
        # 通过user对这个cate的偏好的比例，计算这个cate下的推荐数量
        num = int(ratio * topK) + 1

        if cate not in item_cate:
            continue
        print(cate)
        # 取出这个类别下的推荐item
        recom_list = item_cate[cate][:num]
        print(recom_list)
        recom_result[userid] += recom_list

    return recom_result


if __name__ == '__main__':
    ave_score = read.get_ave_score('../data/ratings.dat')
    item_cate, cate_item = read.get_item_cate('../data/movies.dat', ave_score)

    user_profile = get_user_profile(item_cate, '../data/ratings.dat')

    print(user_profile["1"])
    recom_list = recom(cate_item, user_profile, "1", 10)
    print(recom_list)


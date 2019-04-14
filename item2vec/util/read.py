# -*-coding:utf8-*-
"""
author: Yu
date: 20190414
数据处理
"""
import os


def data2sequence(input_file, output_file):
    """
    将user对item的行为数据转换成item序列数据
    Args:
        input_file: user item rating文件
        output_file: item序列文件保存路径
    Return:
        None
    """
    if not os.path.exists(input_file):
        return

    user_sequence = {}
    score_thr = 4

    with open(input_file) as fb:
        for line in fb:
            record = line.strip().split("::")

            if len(record) != 4:
                print("wrong data: ", line)
                continue

            userid, itemid, rating = record[0], record[1], float(record[2])

            if rating < score_thr:
                continue

            if userid not in user_sequence:
                user_sequence[userid] = []
            user_sequence[userid].append(itemid)

    with open(output_file, 'w') as fb:
        for _, sequence in user_sequence.items():
            fb.write(' '.join(sequence) + '\n')


if __name__ == '__main__':
    data2sequence('../data/ratings.dat', '../data/item_sequences.dat')
    
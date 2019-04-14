# -*-coding:utf8-*-
"""
author: Yu
date: 20190414
使用gensim包下的word2vec模型构建item embedding
"""
from gensim.models import word2vec
import numpy as np


def train_item2vec(input_file, store_path):
    """

    :param input_file:
    :return:
    """

    """
    参数说明：
    sentence：预料，可以是一个列表，也可以使用word2vec提供的语料方法直接从文件中读出
    size：词向量维度，默认是100，小于100M的语料，使用默认值即可，大语料建议增大维度
    window：词向量上下文最大距离，默认值是5，对于一般的语料，推荐值在[5,10]之间
    sg：模型选择，0为CBOW，1为Skip Gram，默认为0
    hs：算法选择，0是Negative Sampling，1是Hierarchical Softmax，默认是0
    negative：使用Negative Sampling算法时，负采样个数，默认是5。推荐在[3,10]之间
    cbow_mean：仅用于CBOW在做投影时，为0表示投影采用和的方式，为1表示投影采用平均值的方式，默认是1
    min_count：需要计算词向量的最小词频，这个值可以去掉一些很生僻的低频词，默认是5。如果语料比较小，建议调低这个值
    iter：随机梯度下降法中迭代的最大次数，默认是5。对于大语料，可以增大这个值
    alpha：随机梯度下降法中迭代的初始步长。默认为0.025
    min_alpha：由于算法支持在迭代的过程中逐渐减小步长，min_alpha给出了最小的迭代步长值。随机梯度下降中每轮的迭代步长可以由iter，alpha， min_alpha一起得出。
    """
    # 从文件中读取语料
    print('Loading data form:', input_file)
    sentences = word2vec.LineSentence(input_file)
    print('Data loading completed. ')

    # 构建模型：128
    print('Start training model...')
    model = word2vec.Word2Vec(sentences=sentences, size=128, sg=1, iter=50, min_count=0)

    print('Saving model to:', store_path)
    model.save(store_path)

    print('Done.')


def get_most_similar(model_path, itemid):
    """
    获取某个item最相似的10个item
    Args:
        model_path: word2vec模型路径
        itemid: 需要推荐的item
    Return:
        a list
    """
    model = word2vec.Word2Vec.load(model_path)

    res = model.wv.most_similar(itemid)

    return res


def item_sim_compute(model_path):
    """
    自行计算所有item的相似度矩阵
    Args:
        model_path: word2vec模型路径
    Return:
        sim: 相似度矩阵
        items2vec: 字典，item到索引到映射
        items: 所有item列表
    """
    model = word2vec.Word2Vec.load(model_path)
    items = model.wv.index2word

    items2index = {}
    M = []
    for index, word in enumerate(items):
        items2index[word] = index
        M.append(model[word])

    M = np.matrix(M)

    # axis=1，表示按行向量求norm
    M_norm = np.linalg.norm(M, axis=1).reshape(len(items), -1)

    # 根据余弦公式进行计算
    sim = (M @ M.T) / (M_norm @ M_norm.T)

    return sim, items2index, items


if __name__ == '__main__':
    # data_path = '../data/item_sequences.dat'
    model_save_path = 'mymodel'
    # train_item2vec(data_path, model_save_path)
    similar = get_most_similar(model_save_path, "1")

    # 可以将sim矩阵保存到数据库
    sim, items2index, items = item_sim_compute(model_save_path)

    index = items2index["1"]
    item_list = sim[index].tolist()[0]
    res = sorted(zip(items, item_list), key=lambda record: record[1], reverse=True)[1:11]

    # 两种获取前10item推荐的结果，几乎没有差别
    print(similar)
    print(res)


# -*-coding:utf8-*-
"""
author: Yu
date: 20190412
LFM model
"""
import numpy as np
from lfm.util import read
import pickle
import os


def init_vector(vector_size):
    """
    initialize a vector
    Args:
        vector_size: len of the vector
    Return:
        a list
    """

    return np.random.randn(vector_size)


def lfm_predict(user_embedding, item_embedding):
    """
    lfm predict result
    Args:
        user_embedding: user vector
        item_embedding: item vector
    Return:
        a float num
    """

    return np.dot(user_embedding, item_embedding)


def lfm_loss(user_vec, item_vec, train_data):
    """
    compute loss
    Args:
        user_vec: user embeddings
        item_vec: item embeddings
        train_data: train data
    Return:
        loss, a num
    """
    m = len(train_data)
    loss = 0

    for data_instance in train_data:
        userid, itemid, label = data_instance
        user_embedding = user_vec[userid]
        item_embedding = item_vec[itemid]

        loss += np.square(label - lfm_predict(user_embedding, item_embedding))

    return loss / m


def lfm_train(train_data, F, alpha, beta, epochs):
    """
    lfm
    Args:
        train_data: train data for lfm
        F: user vector len, item vector len
        alpha: regularization factor
        beta: learning rate
        epochs: iteration num
    Return:
        model loss
    """
    user_vec = {}
    item_vec = {}
    loss_list = []
    for epoch in range(epochs):
        for data_instance in train_data:
            userid, itemid, label = data_instance

            if userid not in user_vec:
                user_vec[userid] = init_vector(F)

            if itemid not in item_vec:
                item_vec[itemid] = init_vector(F)

            delta = lfm_predict(user_vec[userid], item_vec[itemid]) - label

            for index in range(F):
                # 随机梯度下降

                # 梯度更新
                puf = user_vec[userid][index]
                qif = item_vec[itemid][index]
                user_vec[userid][index] -= beta * (delta * qif + alpha * puf)
                item_vec[itemid][index] -= beta * (delta * puf + alpha * qif)

        loss = lfm_loss(user_vec, item_vec, train_data)
        loss_list.append(loss)
        # beta = beta * 0.95

    # user_vec和item_vec存储
    else:
        with open('user_vec_%d_%f_%f_%d.model' % (F, alpha, beta, epochs), 'wb') as fb:
            pickle.dump(user_vec, fb)
        with open('item_vec_%d_%f_%f_%d.model' % (F, alpha, beta, epochs), 'wb') as fb:
            pickle.dump(item_vec, fb)

    return loss_list


def get_recommend_result(user_vec_model, item_vec_model, userid):
    """
    use lfm model result give fix userid recommend result
    Args:
        user_vec_model: user_vec path
        item_vec_model: item_vec path
        userid: fix userid
    Return:
        a list: [(itemid, score), (itemid1, score1)]
    """
    fu = open(user_vec_model, 'rb')
    user_vec = pickle.load(fu)
    fu.close()

    fi = open(item_vec_model, 'rb')
    item_vec = pickle.load(fi)
    fi.close()

    if userid not in user_vec:
        return []

    record = []
    user_embedding = user_vec[userid]
    fix_num = 10

    for itemid in item_vec:
        item_embedding = item_vec[itemid]
        res = np.dot(user_embedding, item_embedding)
        record.append((itemid, round(res, 3)))

    result = sorted(record, key=lambda x: x[1], reverse=True)[:fix_num]
    return result


def ana_recommend_result(train_data, userid, recom_list):
    """
    debug recom result for userid
    Args:
        train_data: train data for userid
        userid: fix userid
        recom_list: recommend result by lfm
    """
    item_info = read.get_item_info('../data/movies.dat')
    for data_instance in train_data:
        tmp_userid, itemid, label = data_instance
        if userid == tmp_userid and label == 1:
            print(itemid, item_info[itemid])
    print('*********' * 20)
    for info in recom_list:
        print(info[0], item_info[info[0]], info[1])


if __name__ == '__main__':
    train_data = read.get_train_data('../data/ratings.dat')
    vec_size_list = [10, 30, 50]
    alpha_list = [0.01, 0.03, 0.1, 0.3, 1]
    beta_list = [0.01, 0.03, 0.1, 0.3, 1]
    epochs_list = [30, 50]

    model_loss = {}

    for vec_size in vec_size_list:
        for alpha in alpha_list:
            for beta in beta_list:
                for epochs in epochs_list:
                    params = "%d_%f_%f_%d" % (vec_size, alpha, beta, epochs)
                    print('train model: ', params)

                    loss_list = lfm_train(train_data, vec_size, alpha, beta, epochs)
                    last_loss = loss_list[-1]

                    print('model: %s, loss: %f .' % (params, last_loss))
                    print(loss_list)

                    if last_loss not in model_loss:
                        model_loss[last_loss] = []
                    model_loss[last_loss].append((params, loss_list))

    with open('model_result.dat', 'wb') as fb:
        pickle.dump(model_loss, fb)

    # recommend_list = get_recommend_result('user_vec.model', 'item_vec.model', '24')
    # ana_recommend_result(train_data, '24', recommend_list)

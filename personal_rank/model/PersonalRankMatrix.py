# -*-coding:utf8-*-
"""
author: Yu
date: 20190413
personal rank的矩阵化计算
"""
from personal_rank.util import mat_util, read
import numpy as np
from scipy.sparse.linalg import lgmres
from scipy.sparse.linalg import inv
import personal_rank.model.PersonalRank as pr


def train_personal_rank(graph, root, alpha):
    """
    矩阵化的方式训练personal rank
    无需迭代计算，一次计算即可完成
    Args:
        graph: 二分图
        root: User
        alpha: 随机游走概率
    Return:
        a dict, key: itemid, value: pr
    """
    M, vertex, address_dict = mat_util.graph2M(graph)

    if root not in address_dict:
        return {}

    mat_all = mat_util.mat_all_point(M, vertex, alpha)

    # 根据公式，现在出初始化一个m+n行1列的r0矩阵
    # 如果要同时计算所有的顶点，只需要将1列拓展为m+n列即可，并将每列的对应位置初始化为1 - alpha
    # 比如这里同时计算了2个顶点
    user_index = address_dict[root]
    initial_list = [[0, 0] for _ in range(len(vertex))]
    initial_list[user_index] = [1 - alpha, 0]
    initial_list[1] = [0, 1 - alpha]

    r0 = np.array(initial_list)

    # 现在的任务是求解第二个公式，也就是mat_all * rank = r0
    # 用到一个稀疏矩阵工具: gmres
    # gmres用来求解形如Ax = b的线性方程问题，但是在实践中发现，b只能是(N,1)的矩阵
    # res = lgmres(mat_all, r0, tol=1e-8)
    # print(res)

    # 可以先求逆，再求解
    # tocsc()指的是转化成csc的格式进行计算
    mat_all_inv = inv(mat_all.tocsc())
    r = (mat_all_inv @ r0).T[user_index]

    return {record[0]: record[1] for record in zip(vertex, r)}


def train_personal_rank_iter(graph, root, alpha, iter_num):
    """
    矩阵化的迭代方式训练personal rank
    实现公式: r = (1 - alpha) + alpha * MT * r
    Args:
        graph: 二分图
        root: User
        alpha: 随机游走的概率
    """
    M, vertex, address_dict = mat_util.graph2M(graph)
    if root not in address_dict:
        return {}

    user_index = address_dict[root]
    # 先初始化一个r，m+n行，1列，如果要同时计算所有的顶点，只需要将1列拓展为m+n列即可，并将每列的对应位置初始化为1
    r = [[0, 0] for _ in range(len(vertex))]
    r[user_index] = [1, 0]
    r = np.matrix(r)
    print(r)

    # 初始化一个r0，，m+n行，1列，如果要同时计算所有的顶点，只需要将1列拓展为m+n列即可，并将每列的对应位置初始化为1 - alpha
    r0 = [[0] for _ in range(len(vertex))]
    r0[user_index] = [1 - alpha]
    r0 = np.matrix(r0)

    for epoch in range(iter_num):
        tmp_r = r0 + alpha * M.todense().transpose() * r

        # 比较两个矩阵所的对应元素是否相等
        if (tmp_r == r).all():
            print('done, iter: %d' % epoch)
            break

        r = tmp_r

    return r.reshape(1, -1).tolist()[user_index]


if __name__ == '__main__':
    # graph = read.get_graph_from_data('../data/ratings.dat')
    # userid = '1'
    # filter_items = graph[userid]
    #
    # rank1 = train_personal_rank(graph, userid, 0.8)
    # res1 = pr.personal_rank_recom(filter_items, rank1, 100)
    # print(res1)
    #
    # rank2 = pr.train_personal_rank(graph, userid, 0.8, 50)
    # res2 = pr.personal_rank_recom(filter_items, rank2, 100)
    # print(res2)
    #
    # count = 0
    # for i in res1:
    #     for j in res2:
    #         if i[0] == j[0]:
    #             count += 1
    #
    # print(count)

    graph = read.get_graph_from_data('../data/test_data')
    r = train_personal_rank(graph, 'A', 0.8)
    print(r)

    print(pr.train_personal_rank(graph, 'A', 0.8, 50))

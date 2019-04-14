# -*-coding:utf8-*-
"""
author: Yu
date: 20190413
矩阵化计算personal rank中用到的矩阵计算
根据公式来的
"""
from scipy.sparse import coo_matrix
import numpy as np
from personal_rank.util import read


def graph2M(graph):
    """
    将二分图转化为M矩阵
    Args:
        graph: 二分图数据
    Return:
        M矩阵，是一个稀疏矩阵
        包含所有节点的列表
        包含节点及其索引的字典
    """
    vertex = list(graph.keys())
    total_len = len(vertex)

    # 保存每个节点对应的index，在进行稀疏矩阵存储的时候需要用到
    address_dict = {}

    for index in range(total_len):
        address_dict[vertex[index]] = index

    row = []
    col = []
    data = []

    # 计算M矩阵，由于需要使用特殊的数据结构来存储M这个稀疏矩阵，所以转换为计算row、col和data
    for element_i in graph:
        # M矩阵中，Mij保存的是i节点的出边数的倒数
        weight = 1.0 / len(graph[element_i])
        row_index = address_dict[element_i]

        for element_j in graph[element_i]:
            col_index = address_dict[element_j]

            # 计算出Mij
            row.append(row_index)
            col.append(col_index)
            data.append(weight)

    # 转化为coo_matrix类型的稀疏矩阵
    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    M = coo_matrix((data, (row, col)), shape=(total_len, total_len))

    return M, vertex, address_dict


def mat_all_point(M, vertex, alpha):
    """
    计算最终公式中的 E - alpha * M.T 的部分
    Args:
        M: M矩阵，是一个稀疏矩阵
        vertex: 包含所有节点的列表
        alpha: 随机游走概率
    Return:
        一个稀疏矩阵，其维度和M相同
    """
    # 先根据节点的索引初始化一个与M矩阵对应的单位矩阵E，同样用稀疏矩阵保存
    total_len = len(vertex)
    row = []
    col= []
    data = []

    for index in range(total_len):
        row.append(index)
        col.append(index)
        data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)

    E = coo_matrix((data, (row, col)), shape=(total_len, total_len))

    return E.tocsr() - alpha * M.tocsr().transpose()


if __name__ == '__main__':
    graph = read.get_graph_from_data('../data/test_data')
    M, vertex, address_index = graph2M(graph)

    res = mat_all_point(M, vertex, 0.8)
    print(res)
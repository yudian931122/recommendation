# -*-coding:utf8-*-
"""
author: Yu
date: 20190413
PersonalRank model
"""
from personal_rank.util import read


def train_personal_rank(graph, root, alpha, iter_num):
    """
    针对某个User训练一个personal rank
    Args:
        graph: 二分图数据
        root: 根节点，也就是需要进行推荐的User
        alpha: 随机游走的概率
        iter_num: 迭代次数
    Return:
        a dict, key itemid, value pr
    """
    # 初始化pr值，root的pr为1，其他节点的pr为0
    rank = {point: 0 for point in graph}
    rank[root] = 1

    for epoch in range(iter_num):
        # 定义一个数据结构，保存本次迭代的结果
        tmp_rank = {point:0 for point in graph}

        # 取出节点i和它的出边尾节点集合ri
        for i, ri in graph.items():
            # 取节点i的出边尾节点j以及边E(i,j)的权重wij，边的权重都为1，在这里实际没有什么作用
            for j, wij in ri.items():
                # 注意，这里要计算的是j的pr值
                # 这时，i是j的一条入边的首节点， len(ri)是i节点的出边数量
                # 公式中，j节点的pr值是根据j的所有入边的首节点的pr贡献值累加而来的
                # 在这里，只计算当前i节点对当前j节点的贡献值，当前j节点的其他入边首节点的贡献值会在之后的循环中累加上来
                tmp_rank[j] += alpha * rank[i] / (1.0 * len(ri))

        # 在公式中表明，root节点除了所有入边的首节点的pr贡献值累加外，还要加上返回到根节点的概率1-alpha
        tmp_rank[root] += 1 - alpha

        # 更新rank
        if rank == tmp_rank:
            print("done, iter: %d" % epoch)
            break

        rank = tmp_rank

    return rank


def personal_rank_recom(filter_items, rank, recom_num):
    """
    根据某个User的rank，给该User推荐制定数量的item
    Args:
        filter_items: User有过行为的item
        rank: 该User的personal rank结果
        recom_num: 推荐item数量
    Return:
        list
    """
    records = filter(lambda record: len(record[0].split("_")) == 2 and record[0] not in filter_items, rank.items())
    records = map(lambda record: (record[0].split("_")[1], record[1]), records)
    records = sorted(records, key=lambda record: record[1], reverse=True)

    if len(records) > recom_num:
        records = records[: recom_num]

    return records


def get_one_user_recom(userid):
    """
    为指定User进行基于图的推荐
    Args:
        userid: 需要推荐的User
    """
    item_info = read.get_item_info('../data/movies.dat')
    graph = read.get_graph_from_data('../data/ratings.dat')
    rank = train_personal_rank(graph, userid, 0.8, 50)

    filter_items = graph[userid]
    for item in filter_items:
        itemid = item.split("_")[1]
        print(item_info[itemid])

    print('*' * 50)

    res = personal_rank_recom(filter_items, rank, 10)
    for item in res:
        print(item_info[item[0]], item[1])


if __name__ == '__main__':
    get_one_user_recom('1')
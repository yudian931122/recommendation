# -*-coding:utf8-*-
"""
author: Yu
date: 20190417
XGBoost
"""
import os
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from scipy.sparse import coo_matrix
from collections import Counter


def data_product(input_file):
    """
    生成XGBoost训练和测试的数据集，数据集使用训练LR模型时使用的数据集
    Args:
        input_file: 数据集路径
    Return:
        train_data: 训练数据集
        test_data: 测试数据集
    """

    if not os.path.exists(input_file):
        print('%s not exists.' % input_file)
        return

    data = pd.read_csv(input_file, header=0)

    # 切分特征和目标
    y = data['y']
    X = data.drop('y', axis=1)

    print(data[data['y'] == 0]['y'].count())
    print(data[data['y'] == 1]['y'].count())

    # 使用过采用平衡正负样本
    smote = SMOTE(random_state=24)
    X, y = smote.fit_sample(X, y)
    print(Counter(y))

    # 数据集切分
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=24, test_size=0.3)

    # 转换xgboost需要的数据格式
    train_data = xgb.DMatrix(X_train, y_train)
    test_data = xgb.DMatrix(X_test, y_test)
    return train_data, test_data, X_train, X_test, y_train, y_test


def train_xgboost_model(X_train, X_test, y_train, y_test):
    """
    网格搜索的方式训练xgboost模型
    Args:
        train_data: 训练数据集
        test_data: 测试数据集
    """
    parameters = {
        'max_depth': [5, 10, 15],
        'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15, 0.3, 0.5],
        'n_estimators': [10, 20, 40],
        'min_child_weight': [0, 2, 5, 10, 20],
        'max_delta_step': [0, 0.2, 0.6, 1, 2],
        'subsample': [0.6, 0.7, 0.8, 0.85, 0.95],
        'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],
        'reg_alpha': [0, 0.25, 0.5, 0.75, 1],
        'reg_lambda': [0.2, 0.4, 0.6, 0.8, 1],
        'scale_pos_weight': [0.2, 0.4, 0.6, 0.8, 1]
    }

    parameters = {
        'max_depth': [4],
        'learning_rate': [0.3],
        'n_estimators': [10]
    }

    xcf = xgb.XGBClassifier()

    grid_search_cv = GridSearchCV(xcf, param_grid=parameters, scoring='roc_auc', cv=5)
    grid_search_cv.fit(X_train, y_train)

    print("Grid Search Best score: %0.3f" % grid_search_cv.best_score_)
    print("Grid Search Best parameters set:")
    estimator_ = grid_search_cv.best_estimator_
    best_parameters = estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    pred = estimator_.predict(X_test)
    auc_score = roc_auc_score(y_test, pred)

    print("Test AUC score: %0.3f" % auc_score)


def xgboost_output_feature(tree_leaf, tree_num, tree_depth):
    """
    将xgboost输出的叶子节点index转化为特征向量
    取单棵树的所有叶子节点组成一个列表，在输出的叶子节点上置1，其他叶子节点上置0
    拼接所有树的叶子节点列表，组成特征
    Args:
        tree_leaf: xgboost输出的叶子节点index，每个样本对应一个list，list长度为tree_num
        tree_num: xgboost模型中树的数量
        tree_depth: xgboost中树的深度
    Return:
        转化后的特征，使用稀疏矩阵进行保存
    """
    # 单棵树中总节点数量
    total_node_num = 2 ** (tree_depth + 1) - 1
    # 单棵树中叶子节点数量
    leaf_node_num = 2 ** tree_depth
    # 单棵树中非叶子节点数量
    not_leaf_node_num = total_node_num - leaf_node_num
    # 转换后的特征的维数
    total_col_num = leaf_node_num * tree_num
    # 样本数
    total_row_num = len(tree_leaf)

    # 定义coo_matrix相关的数据结构
    col = []
    row = []
    data = []

    base_row_index = 0
    for one_result in tree_leaf:
        base_col_index = 0
        for fix_index in one_result:
            # 先减去对非叶子节点的偏移
            leaf_index = fix_index - not_leaf_node_num
            leaf_index = leaf_index if leaf_index >= 0 else 0

            col.append(base_col_index + leaf_index)
            row.append(base_row_index)
            data.append(1)

            # 处理下一颗树的所有叶子节点
            base_col_index += leaf_node_num
        # 处理下一个样本
        base_row_index += 1

    new_feature = coo_matrix((data, (row, col)), shape=(total_row_num, total_col_num))

    return new_feature


def train_xgboost_lr_model(train_data, test_data, X_train, X_test, y_train, y_test):
    """

    :return:
    """
    tree_depth = 4
    paras = {"max_depth": tree_depth, "eta": 0.3}
    tree_num = 10
    # 训练xgboost模型
    bst = xgb.train(paras, train_data, tree_num)

    train_tree_leaf = bst.predict(train_data, pred_leaf=True)
    test_tree_leaf = bst.predict(test_data, pred_leaf=True)

    # 转化特征
    train_feature = xgboost_output_feature(train_tree_leaf, tree_num, tree_depth)
    test_feature = xgboost_output_feature(test_tree_leaf, tree_num, tree_depth)

    lr = LogisticRegressionCV(Cs=[1], cv=5, penalty='l2', max_iter=500, tol=0.0001, scoring='roc_auc')
    lr.fit(train_feature, y_train)
    scores = list(lr.scores_.values())[0]
    print("diff:%s" % (",".join([str(ele) for ele in scores.mean(axis=0)])))
    print("Train AUC:%s(+-%0.2f)" % (scores.mean(), scores.std() * 2))

    pred = lr.predict(test_feature)

    auc_score = roc_auc_score(y_test, pred)
    print("Test AUC:%f0.3" % auc_score)


if __name__ == '__main__':
    data_path = '../data/processed_data_no_fill.csv'
    train_data, test_data, X_train, X_test, y_train, y_test = data_product(data_path)

    # 单独的xgboost和xgboost+lr模型效果对比
    # 在两个xgboost模型使用相同参数的情况下：xgboost+lr效果要优于单独的xgboost
    #
    train_xgboost_model(X_train, X_test, y_train, y_test)
    print("*" * 50)
    train_xgboost_lr_model(train_data, test_data, X_train, X_test, y_train, y_test)

# -*-coding:utf8-*-
"""
author: Yu
date: 20190416
逻辑回归模型实现
"""
import os
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import PolynomialFeatures


def train_lr_model(input_file):
    """
    训练逻辑回归模型
    Args:
        input_file: 输入文件
    Return:

    """
    if not os.path.exists(input_file):
        return

    data = pd.read_csv(input_file, header=0)

    # 查看正负样本数量
    # print(data[data['y'] == 0]['y'].count())
    # print(data[data['y'] == 1]['y'].count())
    """
    对导入的数据集按如下方式进行简单统计可以发现，正样本（y=1）的数量远小于负样本（y=0）的数量，近似等于负样本数量的1/8。
    在分类模型中，这种数据不平衡问题会使得学习模型倾向于把样本分为多数类，但是，我们常常更关心少数类的预测情况。
    在本次分类问题中，分类目标是预测客户是(yes：1)否(no：0)认购定期存款（变量y）。显然，我们更关心有哪些客户认购定期存款。
    为减弱数据不均衡问题带来的不利影响，在数据层面有两种较简单的方法：过抽样和欠抽样。
    1 - 过抽样：
        抽样处理不平衡数据的最常用方法，基本思想就是通过改变训练数据的分布来消除或减小数据的不平衡。
        过抽样方法通过增加少数类样本来提高少数类的分类性能 ，最简单的办法是简单复制少数类样本，缺点是可能导致过拟合，没有给少数类增加任何新的信息，泛化能力弱。
        改进的过抽样方法通过在少数类中加入随机高斯噪声或产生新的合成样本等方法。
    2 - 欠抽样：
        欠抽样方法通过减少多数类样本来提高少数类的分类性能，最简单的方法是通过随机地去掉一些多数类样本来减小多数类的规模，
        缺点是会丢失多数类的一些重要信息，不能够充分利用已有的信息。

    在本次实验中，采用Smote算法[Chawla et al., 2002]增加新的样本进行过抽样；采用随机地去掉一些多数类样本的方法进行欠抽样。
    Smote算法的基本思想是对于少数类中每一个样本x，以欧氏距离为标准计算它到少数类样本集中所有样本的距离，得到其k近邻。
    然后根据样本不平衡比例设置一个采样比例以确定采样倍率N，对于每一个少数类样本x，从其k近邻中随机选择若干个样本，构建新的样本。
    针对本实验的数据，为防止新生成的数据噪声过大，新的样本只有数值型变量真正是新生成的，其他变量和原样本一致。重采样的代码如下：
    """

    y = data['y']
    X = data.drop('y', axis=1)
    print(X.shape)

    # 构建多项式特征
    poly = PolynomialFeatures()
    X = poly.fit_transform(X)

    print(X.shape)
    smote = SMOTE(random_state=24)

    X, y = smote.fit_sample(X, y)
    # 查看SMOTE之后的数据比例
    print(Counter(y))

    # 数据集切分
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=24, test_size=0.3)

    # 训练模型
    # Cs: 正则化系数的倒数
    # cv: k折交叉验证
    # penalty: 正则化类型
    # max_iter: 最大迭代次数
    # tol: 收敛阈值

    # lr = LogisticRegressionCV(Cs=[1], cv=5, penalty='l2', max_iter=500, tol=0.0001)
    # lr.fit(X_train, y_train)
    # scores = list(lr.scores_.values())[0]
    # print("diff:%s" % (",".join([str(ele) for ele in scores.mean(axis=0)])))
    # print("Accuracy:%s(+-%0.2f)" % (scores.mean(), scores.std()*2))

    lr = LogisticRegressionCV(Cs=[1], cv=5, penalty='l2', max_iter=500, tol=0.0001, scoring='roc_auc')
    lr.fit(X_train, y_train)
    scores = list(lr.scores_.values())[0]
    print("diff:%s" % (",".join([str(ele) for ele in scores.mean(axis=0)])))
    print("AUC:%s(+-%0.2f)" % (scores.mean(), scores.std()*2))

    # 预测
    predict = lr.predict(X_test)
    score = roc_auc_score(y_test, predict)
    print(score)


if __name__ == '__main__':
    data_path = '../data/processed_data.csv'
    train_lr_model(data_path)
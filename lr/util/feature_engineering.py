# -*-coding:utf8-*-
"""
author: Yu
date: 20190416
特征工程相关处理
代码主要参考自：http://www.cnblogs.com/llhthinker/p/7101572.html
"""
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


def feature_engineering(input_file, output_file):
    """
    特征工程
    Args:
        input_file: 输入数据集
        output_file: 特征工程完成后，数据集输出路径
    Return:
        None
    """
    if not os.path.exists(input_file):
        return

    data = pd.read_csv(input_file, sep=';', header=0)

    # 通过info信息可以看出，数值型的数据是没有缺失值的
    # 但是根据数据集描述文档中给出的信息，非数值型数据中可能有unknown的未知值
    # 使用如下代码查看字符型变量中unknown值的数量，和缺失值所占的比例
    # for col in data.select_dtypes(include=['object']).columns:
    #     print('unknown value count in %s: %d, unknown ratio: %f' %
    #           (col, data[data[col] == 'unknown']['y'].count(),
    #            data[data[col] == 'unknown']['y'].count() * 100 / data[col].count()))

    # 还有两个需要进行处理的变量
    # duration: 按照数据集文档的建议，在预测过程中应该删除这个变量
    # pdays: 未联系过的用999表示，999和正常值偏离太远，转换为0
    data.drop('duration', axis=1, inplace=True)
    data.loc[data['pdays'] == 999, 'pdays'] = 0

    # 将离散序列变量转换成数值序列，这里指的是education这个变量
    data = encode_edu_attrs(data)

    # 计算特征之间的相关性，主要是针对数值类型的数据
    cor = data.corr()
    cor.loc[:, :] = np.tril(cor, k=-1)  # till将矩阵的上三角变成0
    cor = cor.stack()
    # print(cor[(cor > 0.7) | (cor < -0.7)])

    # 可以看到，在相关性分析中nr.employed, emp.var.rate, euribor3相关性超过了90%
    # 所以选择去除其中两个变量
    data.drop('euribor3m', axis=1, inplace=True)
    data.drop('nr.employed', axis=1, inplace=True)

    # 不同类型的变量进行分类
    numeric_attrs = ['age', 'campaign', 'pdays', 'previous',
                     'emp.var.rate', 'cons.price.idx', 'cons.conf.idx']
    bin_attrs = ['default', 'housing', 'loan']
    cate_attrs = ['poutcome', 'education', 'job', 'marital',
                  'contact', 'month', 'day_of_week']

    # 由上面的统计信息可以看到：
    # job和marital的缺失值较少，分别为330和80。可以选择丢弃这两个字段中存在任意一个字段缺失的样本
    # 如果预计变量对于学习模型效果影响不大，可以对unknown值赋值众数，这里认为变量都对学习模型有较大的影响，不采取此法
    # 可以使用数据完整的行作为训练集，以此来预测缺失值，变量housing，loan，education和default的缺失值采取此法，本次实验使用随机森林预测缺失值
    data = fill_unknown(data, bin_attrs, cate_attrs, numeric_attrs)

    data.to_csv(output_file, index=False)


def fill_unknown(data, bin_attrs, cate_attrs, numeric_attrs):
    """
    缺失值处理
    :param data:
    :param bin_attrs:
    :param cate_attrs:
    :param numeric_attrs:
    :return:
    """
    # fill_attrs = ['education', 'default', 'housing', 'loan']
    fill_attrs = []

    for i in bin_attrs + cate_attrs:
        if data[data[i] == 'unknown']['y'].count() < 500:
            # 缺失值数量小于500的变量，删除存在该变量缺失的样本
            data = data[data[i] != 'unknown']
        else:
            fill_attrs.append(i)

    # 需要先将离散变量数值化之后，才能使用模型来预测缺失值
    # 分类变量又可以分为二项分类变量、有序分类变量和无序分类变量。不同种类的分类变量编码方式也有区别。
    # 在处理的过程中，要注意的是，需要预测的变量的取值和维度（编码）可能会发生变化，在这块代码中，待预测的变量是没有发生维度上的变化的，并且缺失值也依旧为unknown
    data = encode_cate_attrs(data, cate_attrs)
    data = encode_bin_attrs(data, bin_attrs)
    data = trans_bining_attrs(data, ['age'])
    data = trans_num_attrs(data, numeric_attrs)
    data['y'] = data['y'].map({'no': 0, 'yes': 1}).astype(int)

    for i in fill_attrs:
        # 在i变量上缺失的样本
        test_data = data[data[i] == 'unknown']
        # 删除所有待预测的变量，作为测试集
        testX = test_data.drop(fill_attrs, axis=1)

        # 在i变量上没有缺失的样本作为训练集
        train_data = data[data[i] != 'unknown']
        # 本轮需要预测的变量作为目标
        trainY = train_data[i].astype('int')
        # 删除所有待预测的变量，作为训练集
        trainX = train_data.drop(fill_attrs, axis=1)

        # 训练模型并对i进行预测
        predict = train_predict_unknown(trainX, trainY, testX)
        test_data[i] = predict

        data = pd.concat([train_data, test_data])

    return data


def train_predict_unknown(trainX, trainY, testX):
    """
    使用随机森林建立模型，来对缺失值进行预测
    :param trainX:
    :param trainY:
    :param testX:
    :return:
    """
    forest = RandomForestClassifier(n_estimators=100)
    forest.fit(trainX, trainY)

    test_predictY = forest.predict(testX).astype(int)
    return pd.DataFrame(test_predictY, index=testX.index)


def encode_bin_attrs(data, bin_attrs):
    """
    二项分类变量编码
    根据数据集的描述来看，可以认为default，housing和loan是二分类变量，对其进行0,1编码
    Args:
        data: 数据集
        bin_attrs: 二分类变量的名称列表
    Return:
        编码后的数据集
    """
    for i in bin_attrs:
        data.loc[data[i] == 'no', i] = 0
        data.loc[data[i] == 'yes', i] = 1

    return data


def encode_edu_attrs(data):
    """
    根据数据集的描述，可以认为变量education是有序分类变量，影响大小排序为
    "illiterate", "basic.4y", "basic.6y", "basic.9y", "high.school", "professional.course", "university.degree"，
    变量影响又大到小的顺序编码为1,2,3,4,5...
    Args:
        data: 数据集
    Return:
        编码后的数据集
    """
    values = ["illiterate", "basic.4y", "basic.6y", "basic.9y", "high.school", "professional.course",
              "university.degree"]
    levels = range(1, len(values) + 1)
    dict_levels = dict(zip(values, levels))

    for v in values:
        data.loc[data['education'] == v, 'education'] = dict_levels[v]

    return data


def encode_cate_attrs(data, cate_attrs):
    """
    无序分类变量编码，可以认为变量job,marital,contact,month,day_of_week为无序分类变量，
    值得注意的是，虽然变量month,day_of_week从时间的角度是有序的，但是对于目标而言是无序的。
    对于无序分类变量，可以利用哑变量进行编码
    Args:
        data: 数据集
        cate_attrs: 无序分类变量名称列表
    Return:
        编码后的数据集
    """
    data = encode_edu_attrs(data)
    cate_attrs.remove('education')

    for i in cate_attrs:
        dummies_df = pd.get_dummies(data[i])
        # 需要对column重新命名，以免冲突
        dummies_df = dummies_df.rename(columns=lambda x: '_'.join([i, str(x)]))
        data = pd.concat([data, dummies_df], axis=1)
        # 删除原始变量
        data = data.drop(i, axis=1)

    return data


def trans_bining_attrs(data, bining_attrs):
    """
    将连续型特征离散化的一个好处是可以有效地克服数据中隐藏的缺陷： 使模型结果更加稳定。
    例如，数据中的极端值是影响模型效果的一个重要因素。极端值导致模型参数过高或过低，或导致模型被虚假现象"迷惑"，把原来不存在的关系作为重要模式来学习。
    而离散化，尤其是等距离散，可以有效地减弱极端值和异常值的影响。
    Args:
        data: 数据集
        bining_attrs: 需要进行连续型特征离散化的变量名称列表
    Return:
        转换后的数据集
    """
    bining_num = 10

    for attr in bining_attrs:
        data[attr] = pd.qcut(data[attr], bining_num)
        data[attr] = pd.factorize(data[attr])[0] + 1

    return data


def trans_num_attrs(data, numeric_attrs):
    """
    数值变量规范化处理
    Args:
        data: 数据集
        numeric_attrs: 数值类型的变量名称列表
    Return:
        转换后的数据集
    """
    for i in numeric_attrs:
        scaler = StandardScaler()
        data[i] = scaler.fit_transform(data[i].values.reshape(-1, 1))

    return data


if __name__ == '__main__':
    data_path = '../data/bank-additional-full.csv'
    output_path = '../data/processed_data.csv'
    feature_engineering(data_path, output_path)


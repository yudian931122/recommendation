# -*-coding:utf8-*-
"""
author: Yu
date: 20190418
wide and deep 模型
"""
from __future__ import division
import tensorflow as tf
import os
import pandas as pd
from sklearn.utils import shuffle
import numpy as np


def get_feature_column():
    """
    获取wide侧特征和deep侧特征
    数值型特征：age,duration(去除),campaign,pdays,previous,emp.var.rate,cons.price.idx,cons.conf.idx,euribor3m,nr.employed
    离散特征：job,marital,education,default,housing,loan,contact,month,day_of_week,poutcome
    Return:

    """
    age = tf.feature_column.numeric_column("age")
    campaign = tf.feature_column.numeric_column("campaign")
    pdays = tf.feature_column.numeric_column("pdays")
    previous = tf.feature_column.numeric_column("previous")
    emp_var_rate = tf.feature_column.numeric_column("emp.var.rate")
    cons_price_idx = tf.feature_column.numeric_column("cons.price.idx")
    cons_conf_idx = tf.feature_column.numeric_column("cons.conf.idx")
    euribor3m = tf.feature_column.numeric_column("euribor3m")
    nr_employed = tf.feature_column.numeric_column("nr.employed")

    # 离散特征亚编码
    job = tf.feature_column.categorical_column_with_vocabulary_list(
        "job",
        ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services',
         'student', 'technician', 'unemployed']
    )
    marital = tf.feature_column.categorical_column_with_vocabulary_list(
        "marital",
        ['divorced', 'married', 'single']
    )
    education = tf.feature_column.categorical_column_with_vocabulary_list(
        "education",
        ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 'professional.course', 'university.degree']
    )
    default = tf.feature_column.categorical_column_with_vocabulary_list(
        "default",
        ['no', 'yes']
    )
    housing = tf.feature_column.categorical_column_with_vocabulary_list(
        "housing",
        ['no', 'yes']
    )
    loan = tf.feature_column.categorical_column_with_vocabulary_list(
        "loan",
        ['no', 'yes']
    )
    contact = tf.feature_column.categorical_column_with_vocabulary_list(
        "contact",
        ['cellular', 'telephone']
    )
    month = tf.feature_column.categorical_column_with_vocabulary_list(
        "month",
        ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    )
    day_of_week = tf.feature_column.categorical_column_with_vocabulary_list(
        "day_of_week",
        ['mon', 'tue', 'wed', 'thu', 'fri']
    )
    poutcome = tf.feature_column.categorical_column_with_vocabulary_list(
        "poutcome",
        ['failure', 'nonexistent', 'success']
    )

    # 连续特征离散化
    age_bucket = tf.feature_column.bucketized_column(age, [18, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70])
    pdays_bucket = tf.feature_column.bucketized_column(pdays, [5, 10, 15, 20, 25, 30, 35, 40, 45, 50])

    # 组合特征
    cross_feature = [
        tf.feature_column.crossed_column([age_bucket, job], hash_bucket_size=12 * 11),
        tf.feature_column.crossed_column([age_bucket, education], hash_bucket_size=12 * 7)
    ]

    base_columns = [job, marital, education, default, housing, loan, contact, month, day_of_week, poutcome, age_bucket,
                    pdays_bucket]

    # wide侧特征
    wide_columns = base_columns + cross_feature

    # deep侧特征
    deep_columns = [
        age, campaign, pdays, previous, emp_var_rate, cons_price_idx, cons_conf_idx, #euribor3m, nr_employed,
        tf.feature_column.embedding_column(job, 5),
        tf.feature_column.embedding_column(marital, 5),
        tf.feature_column.embedding_column(education, 5),
        tf.feature_column.embedding_column(default, 5),
        tf.feature_column.embedding_column(housing, 5),
        tf.feature_column.embedding_column(loan, 5),
        tf.feature_column.embedding_column(contact, 5),
        tf.feature_column.embedding_column(month, 5),
        tf.feature_column.embedding_column(day_of_week, 5),
        tf.feature_column.embedding_column(poutcome, 5)
    ]

    return wide_columns, deep_columns


def split_train_and_test_data(input_file, output_dir, test_size=0.3):
    """
    切分训练数据和测试数据
    :param input_file:
    :return:
    """
    if not os.path.exists(input_file):
        print('%s not exists.' % input_file)
        return

    data = pd.read_csv(input_file, sep=';', header=0)

    # 去掉含有缺失值的样本
    data = data.replace('unknown', np.nan)
    data = data.dropna(how='any', axis=0)

    print(len(data))

    data = shuffle(data)

    total_size = len(data)

    split = int(total_size * (1 - test_size))

    train_data = data[: split]
    print(len(train_data))
    test_data = data[split:]
    print(len(test_data))

    train_data.to_csv(os.path.join(output_dir, 'train.dat'), index=None)
    test_data.to_csv(os.path.join(output_dir, 'test.dat'), index=None)


def input_fn(data_file, predict, shuffle, re_time, batch_num):
    """
    构造输入数据集
    Args:
        data_file: 数据
        predict: 是否是预测
        shuffle: 是否shuffle
        re_time: 数据集重复次数
        batch_num: 批次大小
    """
    _CSV_COLUMNS_DEFAULT = [0.0, '', '', '', '', '', '', '', '', '', 0.0, 0.0, 0.0, 0.0, '', 0.0, 0.0, 0.0, 0.0, 0.0, '']
    _CSV_COLUMNS = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month',
                    'day_of_week', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'emp.var.rate',
                    'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed', 'y']

    def parse_csv(value):
        """
        解析训练数据
        """
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMNS_DEFAULT)
        features = dict(zip(_CSV_COLUMNS, columns))
        labels = features.pop('y')
        classes = tf.equal(labels, 'yes')
        return features, classes

    def parse_csv_predict(value):
        """
        解析预测数据
        """
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMNS_DEFAULT)
        features = dict(zip(_CSV_COLUMNS, columns))
        return features

    # 从文件中读取数据，跳过第一行
    data_set = tf.data.TextLineDataset(data_file).skip(1)

    if shuffle:
        # 是否进行shuffle
        data_set = data_set.shuffle(buffer_size=30000)
    if predict:
        # 是否是预测数据
        data_set = data_set.map(parse_csv_predict, num_parallel_calls=5)
    else:
        data_set = data_set.map(parse_csv, num_parallel_calls=5)

    # 设置数据集重复次数
    data_set = data_set.repeat(re_time)
    # 设置批大小
    data_set = data_set.batch(batch_num)

    return data_set


def build_model_estimator(wide_columns, deep_columns, model_folder):
    """
    建立wide and deep模型
    Args:
        wide_columns:
        deep_columns:
        model_folder: 模型文件存储路径
    Return:
        model_es: 模型
        serving_input_fn: tf serving输入文件存储路径
    """
    model_es = tf.estimator.DNNLinearCombinedClassifier(
        model_dir=model_folder,
        linear_feature_columns=wide_columns,
        linear_optimizer=tf.train.FtrlOptimizer(0.1, l2_regularization_strength=1.0),
        dnn_feature_columns=deep_columns,
        # 这里能使用FtrlOptimizer吗
        dnn_optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=0.1, l1_regularization_strength=0.001,
                                                        l2_regularization_strength=0.001),
        dnn_hidden_units=[128, 64, 32, 16])

    feature_columns = wide_columns + deep_columns
    feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
    serving_input_fn = (tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec))

    return model_es, serving_input_fn


def train_wide_and_deep_model(train_file, test_file, model_folder, model_export_folder):
    """
    训练wide and deep模型
    Args:
        train_file:
        test_file:
        model_folder: 存放原始模型文件的文件夹
        model_export_folder: 存放供tf serving使用的文件的文件夹
    """
    # 确定wide侧和deep侧的输入特征
    wide_column, deep_column = get_feature_column()

    # 构建wide and deep模型
    estimator, serving_input_fn = build_model_estimator(wide_column, deep_column, model_folder)

    # 训练模型
    res = estimator.train(input_fn=lambda: input_fn(train_file, False, True, 1, 100))
    print(res)
    # 预测, 由于样本不平衡，导致测试结果并不是很理想
    res = estimator.evaluate(input_fn=lambda: input_fn(test_file, False, False, 1, 100))

    print(res)

    # 保存模型文件以及tf serving文件
    estimator.export_savedmodel(model_export_folder, serving_input_fn)


if __name__ == '__main__':
    train_data_path = '../data/train.dat'
    test_data_path = '../data/test.dat'

    train_wide_and_deep_model(train_data_path, test_data_path, '../data/wd', '../data/wd_export')

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/25/025 15:14
# @Author  : zhangyw
# @Site    : 
# @File    : naive_bayes.py
# @Software: PyCharm
import numpy
from sklearn.pipeline import Pipeline

from comm import env,utils
import sklearn
from sklearn import datasets
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer
from sklearn.datasets.base import Bunch
from sklearn import svm

def load_data():
    """
    加载数据
    :return x_train,y_train,x_test,y_test:
    """
    train = datasets.fetch_20newsgroups(subset="train")
    test = datasets.fetch_20newsgroups(subset="test")
    x_train = train.data
    x_test = test.data
    y_train = train.target
    y_test = test.target
    return x_train,y_train,x_test,y_test


def getStopWords():
    """
    获取停用词列表
    :return words:
    """
    with open(utils.Path(paths=[env.datapath,"text"],filename="stop_words.txt").getFilePath()) as file:
        words = file.read().splitlines()
    return words


def classify():
    x_train, y_train, x_test, y_test = load_data()

    nbc = Pipeline([
        ("vect",TfidfVectorizer(stop_words=getStopWords())),
        ("clf",MultinomialNB(alpha=0.01))
    ])

    nbc.fit(x_train,y_train)
    result = nbc.predict(x_test)
    print nbc.score(x_train,y_train)
    print nbc.score(x_test,y_test)


if __name__ == '__main__':
    classify()
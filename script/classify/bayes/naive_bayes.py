#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/25/025 15:14
# @Author  : zhangyw
# @Site    : 
# @File    : naive_bayes.py
# @Software: PyCharm
import numpy

from comm import env,utils
import sklearn
from sklearn import datasets
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets.base import Bunch
from sklearn import svm

def load_data():
    fitbunch = Bunch(data=[],target=[],target_names=[])
    testbunch = Bunch(data=[],target=[],target_names=[])
    print "load news"
    news = datasets.fetch_20newsgroups()
    print "cut datas"
    data = news.data
    print "bulid test datas"
    print "data:",len(data)

    fitbunch.data = data[0:1100]
    fitbunch.target = news.target[0:1100]
    fitbunch.target_names = news.target_names
    return fitbunch

def load_test_data():
    fitbunch = Bunch(data=[], target=[], target_names=[])
    testbunch = Bunch(data=[], target=[], target_names=[])
    print "load news"
    news = datasets.fetch_20newsgroups()
    print "cut datas"
    data = news.data
    print "bulid test datas"
    print "data:", len(data)

    fitbunch.data = data[103:105]
    fitbunch.target = news.target[103:105]
    fitbunch.target_names = news.target_names
    return fitbunch


def do_feature(data):
    vectorizer = TfidfVectorizer()
    tfidt = vectorizer.fit_transform(data.data)
    words = vectorizer.get_feature_names()
    weight = tfidt.toarray()
    result = []

    for index,w in enumerate(weight):
        print "===================第",index,"类文本的权重"
        res = []
        for jndex,h in enumerate(words):
            # print h,w[jndex],data.target[index]
            res.append([h,w[jndex]])
        result.append(res)
    return result,data.target,tfidt


def classify():
    data = load_data()
    features,targets,tfidt = do_feature(data)
    # naive_bayes = MultinomialNB()
    naive_bayes = svm.SVC()
    naive_bayes.fit(tfidt.toarray()[0:1000],targets[0:1000])

    # test_data = load_test_data()
    # test_features,test_targets,test_tfidt = do_feature(test_data)

    res = naive_bayes.predict(tfidt.toarray()[1000:1020])
    print res
    print targets[1000:1020]
    print numpy.array(tfidt.toarray()).shape

classify()
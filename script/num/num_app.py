#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/6/006 15:57
# @Author  : zhangyw
# @Site    : 
# @File    : num_app.py
# @Software: PyCharm
import os

import numpy
from sklearn import svm
from PIL import Image
import pickle
from sklearn.neighbors import KNeighborsClassifier

trainDataPath = "D:\work\code\zhangyingwei\python\py-demos\data\\num\\trainingDigits"
testDataPath = "D:\work\code\zhangyingwei\python\py-demos\data\\num\\testDigits"

def loadData(path = trainDataPath):
    """
    加载数据集
    :param path:
    :return:
    """
    resultDatas = []
    labels = []
    for fileName in os.listdir(path):
        labels.append(fileName.split("_")[0])
        filePath = path+"/"+fileName
        result = []
        with open(filePath) as file:
            lines = file.read().splitlines()
            for line in lines:
                for item in line:
                    result.append(int(item))
        resultDatas.append(result)
    return resultDatas,labels

def loadImage():
    image = Image.open("d://2.png",'r')
    imgArr = numpy.array(image)
    result = []
    for item in imgArr:
        line = []
        for i in item:
            num = sum(i)-765
            if num < 0:
                line.append(1)
            else:
                line.append(num)
        result.append(line)
    return result


def classifyBySvm():
    """
    通过 svm 实现手写数字识别
    :return:
    """
    model = svm.SVC()
    x_train,y_train = loadData(trainDataPath)
    x_test,y_test = loadData(testDataPath)
    print("fit...")
    model.fit(x_train,y_train)
    print("score...")
    score = model.score(x_test,y_test)
    print(score)
    y_result = model.predict(x_test)
    print(y_result)
    print(y_test)
    with open("D:\work\code\zhangyingwei\python\py-demos\data\\num\\num.model","wb") as modelFile:
        pickle.dump(model,modelFile)
        print("dump")

def classifyByKnn():
    x_train, y_train = loadData(trainDataPath)
    x_test, y_test = loadData(testDataPath)
    model = KNeighborsClassifier()
    print("fit...")
    model.fit(x_train,y_train)
    print("score...")
    score = model.score(x_test,y_test)
    print(score)


def doClassify():
    img = loadImage()
    arr = []
    for line in img:
        arr+=line
    with open("D:\work\code\zhangyingwei\python\py-demos\data\\num\\num.model","rb") as modelFile:
        model = pickle.load(modelFile)
    result = model.predict([arr])
    print("识别结果",result)

classifyByKnn()
classifyBySvm()

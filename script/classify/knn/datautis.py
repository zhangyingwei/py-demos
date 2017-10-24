# encoding=utf-8

from comm import env,utils
import urllib2
import numpy

dataurl = "https://github.com/apachecn/MachineLearning/blob/master/input/2.KNN/datingTestSet2.txt"
datapath=utils.Path(paths=[env.datapath,"classify","knn"],filename="datingTestSet").getFilePath()


def readDataAsLines():
    print datapath
    with open(datapath,"r") as file:
        lines = file.read().splitlines()
    return lines


def file2matrix():
    lines=readDataAsLines()
    lines = [line.strip().split("\t") for line in lines]
    numOfLines = len(lines)
    returnMat = numpy.zeros((numOfLines, 3))
    print returnMat
    classLabelVector = []
    for index,line in enumerate(lines):
        # 每列的属性数据
        returnMat[index, :] = line[0:3]
        classLabelVector.append(int(line[-1]))
    return returnMat,classLabelVector
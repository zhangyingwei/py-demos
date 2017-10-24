# encoding=utf-8
import numpy

from comm import env,utils
import sklearn
import datautis
from sklearn import neighbors

knn = neighbors.KNeighborsClassifier()
lines = datautis.readDataAsLines()
lines = [line.strip().split("\t") for line in lines]
labels = [line[3] for line in lines]
lines = [line[0:3] for line in lines]
print lines

knn.fit(lines,labels)
test = ['40920', '8.326976', '0.953952']
print knn.predict(numpy.array([test]))

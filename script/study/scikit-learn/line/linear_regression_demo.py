# encoding=utf-8
import numpy
from comm import env,utils
from sklearn import linear_model

reg = linear_model.LinearRegression()

x = numpy.array([[0, 1, 2],[0, 1, 2],[0, 1, 2]])
y = numpy.array([0, 1, 2])
reg.fit (x, y)

print reg.predict([[0,0,2]])
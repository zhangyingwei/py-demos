# encoding=utf-8
from numpy.ma import array

from comm import env,utils
import matplotlib
import matplotlib.pyplot as plt
import datautis

def drowPoints(dataset,labels):
    fig = plt.figure()
    ax = fig.add_subplot("111")
    ax.scatter(dataset[:,0],dataset[:,1],15.0*array(labels), 15.0*array(labels))
    plt.show()

dataset,labels = datautis.file2matrix()
drowPoints(dataset,labels)
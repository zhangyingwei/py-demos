# encoding=utf-8

from comm import env,utils
import gzip
import pickle
from sklearn import neighbors

"""
手写体识别
"""

def loadData():
    print "load data"
    filepath = utils.Path(paths=[env.datapath,"classify","write_num"],filename="mnist.pkl.gz").getFilePath()
    gz = gzip.open(filepath,"rb")
    training_data, validation_data, test_data = pickle.load(gz)
    gz.close()
    return training_data, validation_data, test_data

def knnTrain():
    training_data, validation_data, test_data = loadData()
    knn = neighbors.KNeighborsClassifier()
    print "fit"
    knn.fit(training_data[0],training_data[1])
    print "predict"
    results = knn.predict(test_data[0][0:100])
    for index,res in enumerate(results):
        print "识别为:"+str(res)
        print "实际为:"+str(test_data[1][0:100][index])
        print "------"

knnTrain()
# loadData()
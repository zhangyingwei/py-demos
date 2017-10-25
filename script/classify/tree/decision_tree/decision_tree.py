# encoding=utf-8

from comm import env,utils
import sklearn
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
import pydot

data = load_iris()


def getTestData():
    return data.data[0:100]

def decisionTreeDemo():
    """
    决策树分类算法识别 鸢尾花 类型
    :return:
    """
    dtree = tree.DecisionTreeClassifier()
    dtree.fit(data.data,data.target)
    testdata = getTestData()
    results = dtree.predict(testdata)
    for index,res in enumerate(results):
        print res,data.target_names[res]

    results2 = dtree.predict_proba(testdata)
    for index,res in enumerate(results2):
        print res,testdata[index]


    dot_data = StringIO()
    tree.export_graphviz(dtree,out_file=dot_data)
    g = pydot.graph_from_dot_data(dot_data.getvalue())
    g[0].write_pdf("iris.pdf")


decisionTreeDemo()
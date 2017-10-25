# encoding=utf-8

from sklearn import datasets
from matplotlib import pyplot as plt


# iris = datasets.load_iris()
# print iris

# from sklearn.datasets import load_digits
# digits = load_digits()
# print(digits.data.shape)
# import matplotlib.pyplot as plt
# plt.gray()
# plt.matshow(digits.images[0])
# plt.show()

# digtis = datasets.load_digits()
# print digtis
#
# print "------"
# linnerud = datasets.load_linnerud()
# print linnerud

news = datasets.fetch_20newsgroups()
print news.data[0:5]
print len(news.target)
print news.target_names
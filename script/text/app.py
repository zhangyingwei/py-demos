# encoding=utf-8

from comm import env
import world_utils
from sklearn.feature_extraction.text import TfidfTransformer #TF-IDF向量转换类
from sklearn.feature_extraction.text import TfidfVectorizer #TF_IDF向量生成类
from sklearn.datasets.base import Bunch
from sklearn.naive_bayes import MultinomialNB

def readWorlds():
    # world_bunch=world_utils.readCutsAsBunch()
    world_bunch=world_utils.readBunchFromDump()
    return world_bunch

def getTFIDFMat(bunch):
    print "getTFIDFMat"
    tfidfspace = Bunch(target_name=bunch.labels, tdm=[],vocabulary={})
    testfidfspace = Bunch(target_name=bunch.labels, tdm=[],vocabulary={})
    vectorizer = TfidfVectorizer(stop_words=world_utils.getStopWordList(), sublinear_tf=True, max_df=0.5)
    transformer = TfidfTransformer()  # 该类会统计每个词语的TF-IDF权值
    tfidfspace.tdm=transformer.fit_transform(vectorizer.fit_transform(bunch.contents))
    tfidfspace.vocabulary = vectorizer.vocabulary_

    # vectorizer2 = TfidfVectorizer(stop_words=world_utils.getStopWordList(), sublinear_tf=True, max_df=0.5,vocabulary=tfidfspace.vocabulary)
    # transformer2 = TfidfTransformer()  # 该类会统计每个词语的TF-IDF权值
    #
    # print bunch.contents[0:10]
    # testfidfspace.tdm = transformer2.fit_transform(vectorizer2.fit_transform(bunch.contents[0:10]))
    # testfidfspace.vocabulary = vectorizer2.vocabulary_
    # testfidfspace.target_name=testfidfspace.target_name[0:10]
    clf = MultinomialNB(alpha=0.001).fit(tfidfspace.tdm, tfidfspace.target_name)
    # predicted = clf.predict(testfidfspace.tdm)
    # print predicted

if __name__ == '__main__':
    bunch=readWorlds()
    getTFIDFMat(bunch)
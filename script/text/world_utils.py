# encoding=utf-8
import os

from comm import env,utils
from sklearn.datasets.base import Bunch
import pickle

text_data_path=env.datapath+"/text/tc-corpus-answer/answer"

def readCutsAsBunch():
    print "read cuts as bunch"
    bunch = Bunch(labels=[],contents=[],filepaths=[])
    for label in os.listdir(text_data_path):
        bunch.labels.append(label)
        filePath=utils.Path(paths=[text_data_path,label,"cut"])
        for fileName in os.listdir(filePath.getFilePath()):
            cutFilePath=utils.Path(paths=[filePath.getFilePath()],filename=fileName)
            bunch.filepaths.append(cutFilePath.getFilePath())
            with open(cutFilePath.getFilePath()) as file:
                words=" ".join(file.read().split(","))
                bunch.contents.append(words)
    print bunch.labels
    return bunch

def dumpBunch(bunch,path=utils.Path(paths=[env.datapath,"text","result"],filename="cuts.data")):
    print "dump bunch"
    savePath = utils.Path(paths=[env.datapath,"text","result"],filename="cuts.data")
    print savePath.getFilePath()
    with open(savePath.getFilePath(),"wb") as file:
        pickle.dump(bunch,file)

def readBunchFromDump():
    print "load bunch"
    savePath = utils.Path(paths=[env.datapath,"text", "result"],filename="cuts.data")
    with open(savePath.getFilePath(),"rb") as file:
        bunch=pickle.load(file)
    return bunch

def getStopWordList():
    with open(utils.Path(paths=[env.datapath,"text"],filename="stop_words.txt").getFilePath(),"r") as file:
        return file.read().splitlines()

if __name__ == '__main__':
    bunch=readCutsAsBunch()
    dumpBunch(bunch)
    # bunch = readBunchFromDump()
    # print bunch.labels
    # print getStopWordList()
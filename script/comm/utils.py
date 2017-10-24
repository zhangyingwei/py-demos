# encoding-utf-8
import os


class Path():
    def __init__(self,paths=[],filename=""):
        self.paths=[str(p) for p in paths]
        self.filename=str(filename)

    def __bulidFilePath__(self):
        p="/".join(self.paths)
        if not os.path.exists(p):
            os.makedirs(p)
        return p

    def getFilePath(self):
        file_path=self.__bulidFilePath__()
        if len(self.filename) == 0:
            return file_path
        else:
            return "/".join([file_path,self.filename])


def printEachArray(arr):
    for a in arr:
        print a
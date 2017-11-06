#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/27/027 15:31
# @Author  : zhangyw
# @Site    : 
# @File    : ngram_demo.py
# @Software: PyCharm

import ngram

model = ngram.NGram(N=3)

# path = "D:/work/code/rzx/liuqianqian/dga/dga_classifier/alexa_100k.txt"
#
# with open(path,"r") as file:
#     lines = file.read().splitlines()
#
# for index,line in enumerate(lines):
#     model.add(line)
#     print "add",index
#
# print model.search("google.com")

model.add("baidu.com")
model.add("baidu1.com")
model.add("baidu2.com")
model.add("baidu3.com")

ress = model.search("ba")

for (_,y) in ress:
    print y
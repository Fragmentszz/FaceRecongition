# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 16:38:18 2018

@author: aoanng
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

from publicmodule import data_tf, data_tf_yb

##创建100个类共10000个样本，每个样本10个特征
train_set = np.array(pd.read_excel("./imgs/Train/traits.xlsx"))
test_set = np.array(pd.read_excel("./imgs/Test/traits.xlsx"))
train_lable = np.array(pd.read_excel("./imgs/Train/labels.xlsx"))
test_lable = np.array(pd.read_excel("./imgs/Test/labels.xlsx"))

train_set, dmean, dstd = data_tf(train_set)
test_set = data_tf_yb(test_set, dmean, dstd)



## 决策树
clf1 = DecisionTreeClassifier(max_depth=50, min_samples_split=2,random_state=0)
clf1.fit(train_set,train_lable.ravel())
# scores1 = cross_val_score(clf1,
# print(scores1.mean())

## 随机森林
clf2 = RandomForestClassifier(n_estimators=20, max_depth=None,min_samples_split=2, random_state=0)
clf2.fit(train_set,train_lable.ravel())
# print(scores2.mean())

## ExtraTree分类器集合
clf3 = ExtraTreesClassifier(n_estimators=20, max_depth=None,min_samples_split=2, random_state=0)
clf3.fit(train_set,train_lable.ravel())
# print(scores3.mean())

res1 = clf1.predict(test_set)
res2 = clf2.predict(test_set)
res3 = clf3.predict(test_set)

print(res1)
print(res2)
count1 = 0
count2 = 0
count3 = 0
for i in range(len(test_lable)):
    if(res1[i] == test_lable[i]):
        count1 += 1
    if(res2[i] == test_lable[i]):
        count2 += 1
    if(res3[i] == test_lable[i]):
        count3 +=1
print((1.0*count1 / len(test_lable),1.0*count2 / len(test_lable),1.0*count3 / len(test_lable)))


res1 = clf1.predict(train_set)
res2 = clf2.predict(train_set)
res3 = clf3.predict(train_set)

print(res1)
print(res2)
count1 = 0
count2 = 0
count3 = 0
for i in range(len(train_lable)):
    if(res1[i] == train_lable[i]):
        count1 += 1
    if(res2[i] == train_lable[i]):
        count2 += 1
    if(res3[i] == train_lable[i]):
        count3 +=1
print((1.0*count1 / len(train_lable),1.0*count2 / len(train_lable),1.0*count3 / len(train_lable)))
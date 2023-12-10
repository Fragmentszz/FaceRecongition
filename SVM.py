import cv2
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import torch.optim as optim
from torch.utils.data import DataLoader

from publicmodule import data_tf, data_tf_yb, GetLoader, MySVM, FaceRecognition

from sklearn.model_selection import train_test_split
from sklearn import svm
train_set = np.array(pd.read_excel("./imgs/Train/traits.xlsx"))
test_set = np.array(pd.read_excel("./imgs/Test/traits.xlsx"))
train_lable = np.array(pd.read_excel("./imgs/Train/labels.xlsx"))
test_lable = np.array(pd.read_excel("./imgs/Test/labels.xlsx"))

train_set, dmean, dstd = data_tf(train_set)
test_set = data_tf_yb(test_set, dmean, dstd)

np.save("dmean", dmean)
np.save("dstd", dstd)

predictor = MySVM(gamma='scale', C=1.0, decision_function_shape='ovr', kernel='linear')
# 进行训练
predictor.fit(train_set, train_lable.ravel())
result = predictor.predict(test_set)
correct = 0
for i in range(test_lable.shape[0]):
    if(result[i] == test_lable[i]):
        correct += 1



print(1.0*correct / test_lable.shape[0])
print(result)
joblib.dump(predictor, './saved_model/svm')

# img = cv2.imread("./imgs/Test/head_Fragments.jpg")
# traits_Vector = np.load("V_final.npy")
# model = predictor
# namelists = pd.read_excel("./namedic.xlsx","Sheet1",index_col=None)
# namedic = []
# for row in namelists["name"]:
#     namedic.append(row)
#     print(row)
# print(namedic)
#
# facerecognition = FaceRecognition(model,traits_Vector,dmean,dstd,namedic)
# res = facerecognition.predict(img)
# cv2.imshow("aa",res)
# cv2.waitKey(0)
res1 = predictor.predict(train_set)
count1 = 0
for i in range(len(train_lable)):
    if(res1[i] == train_lable[i]):
        count1 += 1

print(1.0*count1 / len(train_lable))

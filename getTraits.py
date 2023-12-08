# 该程序用于剪切人脸，并存储为灰度图
import os
from PIL import Image
import getFace
import pandas as pd
import cv2
import numpy as np



def read(str):
    t = 0
    for c in str:
        t = t*10 + (ord(c) - 48)
    return t
def getAllPath(dirpath, *suffix):
    PathArray = []
    labels = []
    for r, ds, fs in os.walk(dirpath):
        for fn in fs:
            val = read(fn[1:4])
            while(len(PathArray) <= val):
                PathArray.append([])
            if os.path.splitext(fn)[1] in suffix:
                fname = os.path.join(r, fn)
                labels.append(val)
                PathArray[val].append(fname)
    return PathArray,labels

def getTraits_Vfinal(rootpath:str):
    filepath = rootpath + "/inited/"
    print(filepath)
    PathArray,labels = getAllPath(filepath,".jpg")
    imageMatrix = []
    for i in range(len(PathArray)):
        count = 0
        if(len(PathArray[i]) == 0):
            continue
        for imgpath in PathArray[i]:
            count += 1
            img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
            ori_size = img.shape
            # 灰度图矩阵
            mats = np.array(img)
            # 将灰度矩阵转换为向量
            imageMatrix.append(mats.ravel())
    imageMatrix = np.matrix(imageMatrix,dtype=float)
    img_mean = np.mean(imageMatrix, axis=1)
    img_mean = img_mean.transpose()
    imageMatrix = imageMatrix.transpose()

    # 去中心化
    imageMatrix = imageMatrix - img_mean
    print(imageMatrix.dtype)
    imag_mat = (imageMatrix.transpose() * imageMatrix)
    imag_mat = imag_mat / imag_mat.shape[0]
    W, V = np.linalg.eig(imag_mat)
    V_img = imageMatrix * V
    axis = W.argsort()[::-1]
    V = V[:,axis]
    V_img = V_img[:, axis]
    number = 0
    x = sum(W)
    traits_count = 0
    for i in range(len(axis)):
        number += W[axis[i]]
        # print(i,number)
        if float(number) / x > 0.9:# 取累加有效值为0.9
            traits_count = i
            print('累加有效值是：', i) # 前62个特征值保存大部分特征信息
            break
    V_final = V_img[:,:traits_count]
    print(V_final.shape)
    traits = imageMatrix.transpose() * V_final

    data = pd.DataFrame(traits)
    data.to_excel(rootpath + "/traits.xlsx","Sheet1",index= False)
    data = pd.DataFrame(labels)
    data.to_excel(rootpath + "/labels.xlsx", "Sheet1",index = False)
    np.save("V_final",V_final)
    return V_final

def getTraitsbyV(rootpath):
    # getFace.readPicSaveFace(rootpath + "/ori",rootpath + "/inited",".jpg")
    PathArray, labels = getAllPath(rootpath + "/inited/",".jpg")
    imageMatrix = []
    V_final = np.load("V_final.npy")
    for i in range(len(PathArray)):
        if(len(PathArray[i]) == 0):
            continue
        for filepath in PathArray[i]:
            img = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
            ori_size = img.shape
            cv2.imshow("a",img)
            # 灰度图矩阵
            mats = np.array(img)
            imageMatrix.append(mats.ravel())
    imageMatrix = np.matrix(imageMatrix, dtype=float)

    print(imageMatrix.shape)
    traits = imageMatrix * V_final
    data = pd.DataFrame(traits)
    data.to_excel(rootpath + "/traits.xlsx", "Sheet1",index = False)
    data = pd.DataFrame(labels)
    data.to_excel(rootpath + "/labels.xlsx", "Sheet1",index = False)



if __name__ == '__main__':
    V_final = getTraits_Vfinal("./imgs/Train")
    getTraitsbyV("./imgs/Test")
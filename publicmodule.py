import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import pandas as pd
import cv2
import getFace
# 矩阵标准化
def data_tf(data):
    dmean = np.mean(data,axis=0)
    dstd = np.std(data, axis=0)
    data -= dmean
    data /= dstd
    return data,dmean,dstd

def data_tf_yb(data,dmean,dstd):
    data -= dmean
    data /= dstd
    return data


# 定义GetLoader类，继承Dataset方法，并重写__getitem__()和__len__()方法
class GetLoader(torch.utils.data.Dataset):
    # 初始化函数，得到数据
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label

    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index][0]
        return data, labels

    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)




class BPNNModel(nn.Module):
    def __init__(self):
        # 调用父类的初始化函数，必须要的
        super(BPNNModel, self).__init__()
        # 创建四个Sequential对象，Sequential是一个时序容器，将里面的小的模型按照序列建立网络
        self.layer1 = nn.Sequential(nn.Linear(76, 200), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(200, 350), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Linear(350, 500), nn.ReLU())
        self.layer4 = nn.Sequential(nn.Linear(500, 704))

    def forward(self, img):
        # 每一个时序容器都是callable的，因此用法也是一样。
        img = self.layer1(img)
        img = self.layer2(img)
        img = self.layer3(img)
        img = self.layer4(img)
        return img


class FaceRecognition:
    Model = None
    traits_vector = None
    dmean = None
    dstd = None
    namedic = None
    def __init__(self,model,_traitsV,_dmean,_dstd,_namedic):
        self.Model = model
        self.dmean = _dmean
        self.dstd = _dstd
        self.traits_vector = _traitsV
        self.namedic = _namedic
    def predict(self,img):
        img_ori = img
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = getFace.getFace(img)
        img_result = img_ori
        print(len(faces))
        for img2,x1,y1,x2,y2 in faces:
            print(x1,y1,x2,y2)
            imgvector = np.matrix(np.array(img2).ravel())
            traits = imgvector * self.traits_vector
            print(traits.shape)
            traits = data_tf_yb(traits,self.dmean,self.dstd)
            self.Model.eval()  # 将模型改为预测模式
            x = torch.from_numpy(traits).type(torch.FloatTensor)
            out = self.Model(x)
            _, prediction = torch.max(out, 1)
            # 将预测结果从tensor转为array，并抽取结果
            prediction = prediction.numpy()[0]
            print(prediction)
            img_result = cv2.rectangle(img_result,(x1,y1),(x2,y2),(0,0,255))
            img_result = cv2.putText(img_result,self.namedic[int(prediction)],(x1,y1),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale= 0.75,color=(0,0,255))

        return img_result

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import Train
import getFace
from torch.autograd import Variable
from publicmodule import *



if __name__ == '__main__':
    img = cv2.imread("./imgs/Test/ori/i025qa-mn.jpg")
    model = BPNNModel()
    model.load_state_dict(torch.load("./bp_model"))
    model.eval()
    namelists = pd.read_excel("./namedic.xlsx","Sheet1")
    namedic = {}
    for namelist in namelists:
        namedic[namelist[0]] = namelist[1]
    print(namedic)
    traits_Vector = np.load("V_final.npy")
    dstd = np.load("dstd.npy")
    dmean = np.load("dmean.npy")
    facerecognition = FaceRecognition(model,traits_Vector,dmean,dstd,namedic)
    res = facerecognition.predict(img)
    cv2.imshow("aa",res)
    cv2.waitKey(0)

    # data = np.matrix([[1,2,3],[3,25,3]])
    # print(np.mean(data,axis=1))
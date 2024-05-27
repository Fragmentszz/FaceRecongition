import PIL.Image
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
import PIL
from PIL import Image
shape0 = 64

cell_size = 16
bin_size = 9
every_angle = 180 / (bin_size)
block_size = 2

def getdim():
    return int((shape0 //  cell_size) - block_size + 1)  * int((shape0 // cell_size) - block_size + 1) *  block_size * block_size * bin_size

def getGradiantDirection(img):
    # 计算图像的梯度
    # 计算x方向和y方向的梯度
    grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    # 计算梯度的大小和方向
    magnitude, angle = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)
    return magnitude, angle
def getHOG_train(img,cato):
    img = np.array(img,dtype=np.float32)
    # img = img / 256
    M, A = getGradiantDirection(img)
    A = A % 180

    bin = np.zeros((img.shape[0] // cell_size, img.shape[1] // cell_size, bin_size))
    
    for i in range(0, img.shape[0] - cell_size + 1, cell_size):
        for j in range(0, img.shape[1] - cell_size + 1, cell_size):
            window = A[i:i + cell_size, j:j + cell_size]
            m_window = M[i:i + cell_size, j:j + cell_size]
            window = np.abs(window)
            for k1 in range(cell_size):
                for k2 in range(cell_size):
                    w = int(window[k1,k2] // every_angle)
                    if w == bin_size:
                        w = 0
                    bin[i // cell_size, j // cell_size, w] += m_window[k1,k2] * (1 - (np.abs(window[k1,k2] - w * every_angle) / every_angle))
                    bin[i // cell_size, j // cell_size, (w + 1) % bin_size] += m_window[k1,k2] * (np.abs(window[k1,k2] - w * every_angle) / every_angle)
    # print(img.shape)
    # print(bin.shape)
    traits = np.zeros(((bin.shape[0] - block_size + 1)*(bin.shape[1] - block_size + 1)*(block_size * block_size * bin_size) + 1))
    now = 0
    per_block = block_size * block_size * bin_size
    for i in range(0, bin.shape[0] - block_size + 1):
        for j in range(0, bin.shape[1] - block_size + 1):
            block = bin[i:i + block_size, j:j + block_size, :]
            block = block.ravel()
            if(np.linalg.norm(block) > 1e-6):
                block = block / np.linalg.norm(block)
            else:
                block = np.zeros((block_size * block_size * bin_size))
            traits[now:now+per_block] = block
            now += per_block
    traits[now] = cato
    traits = traits.ravel()

    return traits
def getHOG(oriimg):
    img = np.array(oriimg,dtype=np.float32)
    
    M, A = getGradiantDirection(img)
    A = A % 180

    bin = np.zeros((img.shape[0] // cell_size, img.shape[1] // cell_size, bin_size))
    
    for i in range(0, img.shape[0] - cell_size + 1, cell_size):
        for j in range(0, img.shape[1] - cell_size + 1, cell_size):
            window = A[i:i + cell_size, j:j + cell_size]
            m_window = M[i:i + cell_size, j:j + cell_size]
            window = np.abs(window)
            for k1 in range(cell_size):
                for k2 in range(cell_size):
                    w = int(window[k1,k2] // every_angle)
                    if w == bin_size:
                        w = 0
                    bin[i // cell_size, j // cell_size, w] += m_window[k1,k2] * (1 - (np.abs(window[k1,k2] - w * every_angle) / every_angle))
                    bin[i // cell_size, j // cell_size, (w + 1) % bin_size] += m_window[k1,k2] * (np.abs(window[k1,k2] - w * every_angle) / every_angle)
    # print(img.shape)
    # print(bin.shape)
    traits = np.zeros(((bin.shape[0] - block_size + 1)*(bin.shape[1] - block_size + 1)*(block_size * block_size * bin_size)))
    now = 0
    per_block = block_size * block_size * bin_size
    for i in range(0, bin.shape[0] - block_size + 1):
        for j in range(0, bin.shape[1] - block_size + 1):
            block = bin[i:i + block_size, j:j + block_size, :]
            block = block.ravel()
            if(np.linalg.norm(block) > 1e-6):
                block = block / np.linalg.norm(block)
            else:
                block = np.zeros((block_size * block_size * bin_size))
            traits[now:now+per_block] = block
            now += per_block
    traits = traits.ravel()

    return traits
if __name__ == '__main__':
    datas = []

    for img in os.listdir("/gpfs/home/P02114015/faceReconition/imgs/positive"):
        # img = cv2.imread("/gpfs/home/P02114015/faceReconition/imgs/positive"+img,cv2.IMREAD_GRAYSCALE)
        print(img)
        img = Image.open("/gpfs/home/P02114015/faceReconition/imgs/positive/"+img)

        # 转换为灰度图像
        try:
            img = img.convert("L")
        except ValueError as e:
            print(f"Error: {e}")
        
        img = img.resize((shape0,int(shape0*(img.size[1]/img.size[0]))))
        traits = getHOG_train(img,1)
        np.save("./traits",traits)
        # print(traits)
        
        datas.append(traits)

    print("positive over")
    for img in os.listdir("/gpfs/home/P02114015/faceReconition/imgs/negtive2"):
        if np.random.uniform(0,1,(1)) >= 0.025:
            continue
        img = Image.open("/gpfs/home/P02114015/faceReconition/imgs/negtive2/"+img)
        # 转换为灰度图像
        try:
            img = img.convert("L")
        except ValueError as e:
            print(f"Error: {e}")
        img = img.resize((shape0,int(shape0*(img.size[1]/img.size[0]))))
        traits = getHOG_train(img,0)
        datas.append(traits)
    columns =  []
    print("negtive over")
    for i in range(len(datas[0])-1):
        columns.append('x' + str(i))
    columns.append('y')

    dataframe = pd.DataFrame(datas,columns=columns)
    dataframe.to_excel("/gpfs/home/P02114015/faceReconition/HOG/traits.xlsx",index=False)
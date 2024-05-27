import math

import PIL
import joblib
import pandas as pd
from PIL import Image
from HOG import getHOG,getdim
import numpy as np
import cv2
model_path = "/gpfs/home/P02114015/faceReconition/FaceDetaction/HOG_detaction.pkl"
model = joblib.load(model_path)
columns = []
data = []

dim = getdim()

for i in range(dim):
    data.append(i)
    columns.append('x' + str(i))
x = pd.DataFrame(data=[data],columns=columns)
def findFaces_(img)->list:
    w=64
    h=64
    imgw,imgh = img.size
    aimg = np.mat(img.copy(),dtype=np.float32)
    faces = []
    bc = 64
    for i in range(0,imgw-w + 1,10):
        for j in range(0,imgh-h + 1,10):
            window = aimg[i:i+w,j:j+h]
            if(window.shape != (64,64)):
                continue
            feature = getHOG(window).reshape(1,-1)

            x.iloc[:,:] = feature
            y = model.predict(x)
            if(y[0] == 1):
                faces.append((i,j))
                bc = 64
            else:
                bc = bc * 2
            # i += bc
    return faces


def findFaces(img_path:str):
    start = -2
    last = 1
    shape0 = 128
    img = Image.open(img_path)
    oriimg = cv2.imread(img_path)
    try:
        img = img.convert("L")
    except ValueError as e:
        print(f"Error: {e}")
        return []
    img = img.resize((shape0,int(shape0*(img.size[1]/img.size[0]))))
    bl0 = oriimg.shape[1] / shape0
    print(bl0)
    faces = []
    for scale in range(start,last+1):
        bl = math.sqrt(2) ** scale
        print(bl,scale)
        newimg = img.copy()
        newimg = newimg.resize((int(img.size[0]*bl),int(img.size[1]*bl)))
        faces_t = findFaces_(newimg)
        for face in faces_t:
            faces.append((face[0] * 1 / bl,face[1] * 1 / bl,64.0 / bl,64.0 /bl))
    for (x, y, w, h) in faces:
        x = int(x*bl0)
        y = int(y*bl0)
        w = int(w*bl0)
        h = int(h*bl0)
        cv2.rectangle(oriimg, (y,x), (y + h,x + w), (0, 0, 255), 2)
    print(faces)
    return oriimg
    
    # cv2.imshow("a",img)
    # cv2.imwrite("./result.jpg",img)


if __name__ == "__main__":
    img_path = "/gpfs/home/P02114015/faceReconition/imgs/self/i700_12.jpg"

    findFaces(img_path)
    # cv2.waitKey(0)
    # shape0 = 640
    # img = Image.open(img_path)
    # try:
    #     img = img.convert("L")
    # except ValueError as e:
    #     print(f"Error: {e}")
    # img = img.resize((shape0, int(shape0 * (img.size[0] / img.size[1]))))
    # img = np.array(img)
    # print(getHOG(img[:64,:64]).shape)
    # print(model.predict(getHOG(img[:64,:64]).reshape(1,-1)))



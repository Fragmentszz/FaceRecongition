import cv2
import random
import os
XMLpath = "/gpfs/home/P02114015/faceReconition/FaceDetaction/haarcascade_frontalface_alt2.xml" # 级联分类器的地址，换成自己的


def random_crop(image_path, crop_width, crop_height):
    """
    从给定路径的图像中随机裁剪一个指定大小的窗口。

    参数:
        image_path (str): 图像文件的路径。
        crop_width (int): 裁剪窗口的宽度。
        crop_height (int): 裁剪窗口的高度。

    返回:
        cropped_image (numpy.ndarray): 裁剪后的图像。
    """
    # 读取图像
    img = cv2.imread(image_path)
    img_height, img_width, _ = img.shape

    # 确保裁剪窗口的大小不超过图像的大小
    if crop_width > img_width or crop_height > img_height:
        raise ValueError("裁剪窗口的大小不能超过图像的大小")

    # 随机选择裁剪窗口的左上角位置
    left = random.randint(0, img_width - crop_width)
    top = random.randint(0, img_height - crop_height)

    # 裁剪图像
    cropped_image = img[top:top + crop_height, left:left + crop_width]

    return cropped_image

if __name__ == "__main__":
    face_cascade = cv2.CascadeClassifier(XMLpath)
    imgpath = "/gpfs/home/P02114015/faceReconition/imgs/Train/ori"
    # print(os.listdir(imgpath))
    w,h = 200,200
    ev = 20
    cnt1 = 0
    cnt2 = 0
    for img in os.listdir(imgpath):
        img_path = imgpath + "/" + img
        if(cnt1 > 0):
            break
        print(img_path)
        for k in range(ev):
            nowimg = random_crop(imgpath + "/" + img,w,h)
            faces = face_cascade.detectMultiScale(nowimg)
            if(len(faces) != 0):
                cv2.imwrite("/gpfs/home/P02114015/faceReconition/imgs/positive" + "/" + "positive_" + str(cnt1) + ".jpg",nowimg)
                cnt1+=1
            else: 
                cv2.imwrite("/gpfs/home/P02114015/faceReconition/imgs/negtive2" + "/" + "negtive_" + str(cnt2) + ".jpg",nowimg)
                cnt2+=1    
    print(cnt1,cnt2)

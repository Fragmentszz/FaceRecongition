from flask import Flask, request, send_from_directory, render_template, redirect, url_for,jsonify
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from findFaces import findFaces
app = Flask(__name__)

# 配置上传文件夹
UPLOAD_FOLDER = '/gpfs/home/P02114015/faceReconition/Server/uploads'
PROCESSED_FOLDER = '/gpfs/home/P02114015/faceReconition/Server/static/processed'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# 确保上传和处理文件夹存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 处理图片
        processed_image_path = process_image(filepath, filename)
        return jsonify({'path': url_for('display_image', filename=os.path.basename(processed_image_path))})


@app.route('/processed/<filename>')
def display_image(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)



def process_image(filepath, filename):
    # # 读取图像
    # image = cv2.imread(filepath)
    
    # # 加载人脸检测模型
    # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # # 转换为灰度图像
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # # 检测人脸
    # faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # # 绘制矩形框
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # 保存处理后的图像
    img = findFaces(filepath)
    
    processed_image_path = os.path.join(app.config['PROCESSED_FOLDER'], 'processed_' + filename)
    print(processed_image_path)
    cv2.imwrite(processed_image_path, img)
    
    return processed_image_path

if __name__ == '__main__':
    app.run(debug=True)

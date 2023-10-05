from flask import Flask, request, jsonify, render_template
import joblib
import cv2
import os
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html')


@app.route('/predict',methods = ['GET','POST'])
def result():
    s=[]

    def img_to_array(filename): 
        gray = cv2.imread(filename,cv2.IMREAD_GRAYSCALE) 
        flat_image = gray.flatten()
        return flat_image
    # image = request.form['image']
    size = request.form['radio_4']
    algo = request.form['radio_5']
    image_path  = ''
    # if size == '(8,8)':
    #     if algo == 'SVM':
    #         model = joblib.load('svm_model_64.pkl')
    #         predict = model.predict(image)
    #     elif algo == 'KNN':
    #         model = joblib.load('knn_model_64.pkl')
    #         predict = model.predict(image)
    #     else:
    #         model = joblib.load('dt_model_64.pkl')
    #         predict = model.predict(image)
    if 'imageFile' not in request.files:
        return "No file part"
# 
    file = request.files['imageFile']

    if file.filename == '':
        return "No selected file"
    image = ''
    if file:
        image_path = os.path.join(r'C:\Users\kvkak\Desktop\Minor_dataset\original_dataset\1201',file.filename)
        image = cv2.imread(image_path)
    if size == '(8,8)':
        size = (8,8)
    elif size == '(16,16)':
        size = (16,16)
    else:
        size = (32,32)
    print(size)
    image = cv2.resize(image,size)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    fg = cv2.bitwise_and(gray, gray, mask=mask)
    image_array = fg.flatten()
    test_image = image_array
    test = np.array(test_image)
    if size== (8,8):
        test = np.reshape(test,(1,64))
    elif size == (16,16):
        test = np.reshape(test,(1,256))
    else:
        test = np.reshape(test,(1,1024))

    data =  {
        'size': size,
        'algo': 'algo'
    }
    return render_template('result.html',data = data)


if __name__ == '__main__':
    app.run(debug=True, port= 3000)
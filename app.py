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
    def print_names(n):
        if n==1:
            return 'Akanksha'
        elif n==2:
            return 'Mouli'
        elif n==8:
            return 'Prajay'
        elif n==12:
            return 'Divya'
        elif n==14:
            return 'Sravani'
        elif n==17:
            return 'Chandana'
        elif n==21:
            return 'Vaishnavi'
        elif n==22:
            return 'Sanjana'
        elif n==24:
            return 'Suprathika'
        elif n==26:
            return 'Vamshi'
        elif n==29:
            return 'Tejaswini'
        elif n==31:
            return 'Abhilasha'
        elif n==32:
            return 'Sai Teja'
        elif n==33:
            return 'Manish'
        elif n==35:
            return 'Bhuvaneshwari'
        elif n==36:
            return 'Pranav Karthik'
        elif n==37:
            return 'Mukesh'
        elif n==41:
            return 'Pranay'
        elif n==42:
            return 'Rasool'
        elif n==44:
            return 'Rukmini'
        elif n==59:
            return 'Adarsh'
        elif n==60:
            return 'Pranitha'
        
    
    # image = request.form['image']
    size = request.form['radio_4']
    algo = request.form['radio_5']
    print('Size',size)
    print('Algo',algo)
    image_path  = ''
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
    test = np.array(image_array)
    if size== (8,8):
        test = np.reshape(test,(1,64))
    elif size == (16,16):
        test = np.reshape(test,(1,256))
    else:
        test = np.reshape(test,(1,1024))
    if size == (8,8):
        if algo == 'SVM':
            model = joblib.load('svm_model_64.pkl')
            predict = model.predict(test)
        elif algo == 'KNN':
            model = joblib.load('knn_model_64.pkl')
            predict = model.predict(test)
        else:
            model = joblib.load('dt_model_64.pkl')
            predict = model.predict(test)
    elif size == (16,16):
        if algo == 'SVM':
            model = joblib.load('svm_model_256.pkl')
            predict = model.predict(test)
        elif algo == 'KNN':
            model = joblib.load('knn_model_256.pkl')
            predict = model.predict(test)
        else:
            model = joblib.load('dt_model_256.pkl')
            predict = model.predict(test)
    else:
        if algo == 'SVM':
            model = joblib.load('svm_model_1024.pkl')
            predict = model.predict(test)
        elif algo == 'KNN':
            model = joblib.load('knn_model_1024.pkl')
            predict = model.predict(test)
        else:
            model = joblib.load('dt_model_1024.pkl')
            predict = model.predict(test)
    print('',predict)
    res = print_names(predict[0])
    print('Result',res)
    data =  {
        'Result':res,
        'size': size,
        'algo': 'algo'
    }
    return render_template('result.html',data = data)


if __name__ == '__main__':
    app.run(debug=True, port= 3000)
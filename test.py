from flask import Flask, request
import numpy as np
import cv2
import requests
import os


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def hello():
    return 'Hello, This is YWJC API!'

@app.route('/ywjc', methods=['GET', 'POST'])
def process_image():
    global detector
    if request.json:
        data = request.json
        print(data)
        image_path = data.get("image_path")
        pic = cv2.imread(image_path)
        result = [[1],[2]]
        if result is None: 
            return str('{"response": "image cannot open!"}')
        
        result_path = image_path.split('.')[0] + '_result.jpg'
        
        have_yw = "yw" if len(result)>0 else "myw"
        result_str = str('{"response": "%s %s"}'%(have_yw, result_path))
        return result_str
    else:
        return "no json received"


if __name__ == "__main__":
    app.run(host='127.0.0.1', debug=False, port=8080)
import os
import time
import json
from logger import _print

import cv2
from flask import Flask, request

from Detector import Detector


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def hello():
    return 'Hello, This is YWJC API!'

    
@app.route('/ywjc', methods=['GET', 'POST'])
def process_image():
    #global detector
    if request.json:
        data = request.json
        _print("Request: %s"%data)
        image_path = data.get("image_path")
        
        pic = cv2.imread(image_path)
        _print("Loading image : {}".format(image_path))
        start_time = time.time()
        result = detector.detect(image_path, keep_best=False)
        _print("Detect Using : {:.2f} seconds".format((time.time() - start_time)))
        _print(result)
        
        if result is None:
            _print("{} image cannot open!".format(image_path))
            return json.dumps('{"response": "image cannot open!"}')      
        _print("%s objs found!"%len(result))
        result_path = image_path.split('.')[0] + '_result.jpg'
        detector.save(result_path)   
        
        have_yw = "yw" if len(result)>0 else "myw"
        response = {"response": "%s %s"%(have_yw, result_path.replace("\\", "/"))}
        _print("Response: %s"%response)
        _print("*-"*40 + "\n")
        return json.dumps(response)
    else:
        return json.dumps("no json received")


if __name__ == "__main__":
    _print("Start detection!")
    detector = Detector(config_file="E:/DL/det2/YJH/retinanet_R_50_FPN_1x.yaml", class_names=['nest', 'floating'])
    _print("*-"*40 + "\n")
    app.run(host='127.0.0.1', debug=False, port=7890)

import os
from flask import Flask, render_template, request
import cv2
import numpy as np
import base64

from Detector import Detector

app = Flask(__name__)

UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def start_page():
    print("Start")
    return render_template('index.html')

    
@app.route('/upload', methods=['POST'])
def upload_file():
    global detector
    file = request.files['image']

    # Save file
    #filename = 'static/' + file.filename
    #file.save(filename)

    # Read image
    image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    
    objs = detector.detect(image)
    num_yw = len(objs)
    
    # Save
    cv2.imwrite('result.jpg', detector.image)
    
    # In memory
    image_content = cv2.imencode('.jpg', detector.image)[1].tostring()
    encoded_image = base64.encodestring(image_content)
    to_send = 'data:image/jpg;base64, ' + str(encoded_image, 'utf-8')

    return render_template('index.html', ywDetected=(len(objs) > 0), num_yw=num_yw, image_to_show=to_send, init=True)


if __name__ == "__main__":
    detector = Detector(class_names=['nest', 'floating'])
    # Only for debugging while developing
    app.run(host='0.0.0.0', debug=False, port=7890)
    
    
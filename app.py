from flask import Flask, request, jsonify
from PIL import Image
import io
import glob
from ultralytics import YOLO
import base64
import numpy as np
from utils.inference import Inference
from flask_cors import CORS
import cv2

app = Flask(__name__)
CORS(app)

# Define the inference function (example, modify based on your model)
def make_inference(image):
    infer = Inference(image, names, feats)
    meta = infer.meta_data(detection_model_faces, detection_model_flag, classification_model)

    return meta

@app.route('/upload-image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Open the image file

        # Call the inference function with the image
        image = Image.open(file.stream)
        
        result = make_inference(image)
        print(result)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    classification_model = YOLO(".\\weights\\FlagCls.pt")  # load a pretrained model (recommended for training)
    detection_model_flag = YOLO('.\\FaceFlagDetection.pt')#FaceFlagDetection.pt')
    detection_model_faces = YOLO('.\\yolov8l-face.pt')#FaceFlagDetection.pt')
    loaded_data = np.load('.\\face_features.npz')
    names = loaded_data['Names']
    feats = loaded_data['Features'] 
    app.run(debug=True, port=5000)

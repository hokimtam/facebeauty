from flask import Flask, request, jsonify
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load Haarcascade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

def estimate_smile_intensity(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
        
        if len(smiles) > 0:
            smile_width = max([sw for (sx, sy, sw, sh) in smiles])
            smile_intensity = min(100, int((smile_width / w) * 100))
            
            if smile_intensity <= 20:
                category = "No Smile"
            elif smile_intensity <= 40:
                category = "Mild Smile"
            elif smile_intensity <= 60:
                category = "Grin"
            elif smile_intensity <= 80:
                category = "Big Smile"
            else:
                category = "Laughing"
            
            return {"smile_intensity": smile_intensity, "category": category}
    
    return {"smile_intensity": 0, "category": "No Smile"}

@app.route('/detect_smile', methods=['POST'])
def detect_smile():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    result = estimate_smile_intensity(filepath)
    os.remove(filepath)
    
    return jsonify(result)

@app.errorhandler(403)
def forbidden_error(error):
    return jsonify({"error": "Forbidden access", "details": str(error)}), 403

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5003)

from flask import Flask, request, jsonify
from keras.models import load_model
from PIL import Image
import numpy as np
import io
import os
from ultralytics import YOLO
import pickle
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from io import BytesIO
import pickle
app = Flask(__name__)

# Load the ResNet50V2 model
model_path = 'ResNet50V2.h5'  # Update with your model path
resnet_model = load_model(model_path)
# File to store known faces
FACE_DATA_FILE = "face_encodings.pkl"

# Define the class names for ResNet50V2
class_names = [
    '1000Rs', '1000Rsback', '100Rs', '100Rsback', '10Rs', '10Rsback',
    '20Rs', '20Rsback', '5000Rs', '5000Rsback', '500Rs', '500Rsback',
    '50Rs', '50Rsback'
]

# Load the YOLO model
# yolo_model = YOLO("yolov8x-oiv7.pt")  # Load a pretrained model
yolo_model = YOLO("yolov8n.pt")  # Load a pretrained model

@app.route('/hello', methods=['GET'])
def hello():
    return 'Hello, World!'

@app.route('/currencyprediction', methods=['POST'])
def currencyprediction():
    # Check if the request contains a file
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    try:
        # Convert image to bytes and open with PIL
        image_data = file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Preprocess the image
        image = image.resize((224, 224))  # Resize to match the model input
        image_array = np.array(image)  # Convert to NumPy array
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        image_array = image_array / 255.0  # Normalize pixel values
        
        # Perform inference
        predictions = resnet_model.predict(image_array)
        max_confidence = np.max(predictions)
        predicted_class = class_names[np.argmax(predictions)]
        
        # Apply threshold
        if max_confidence < 0.7:
            return jsonify({'predictions': 'No note detected'})
        
        # Return the prediction
        return jsonify({'predictions': predicted_class})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# @app.route('/predict', methods=['POST'])
# def predict():
#     # Check if the request contains a file
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file provided'}), 400
    
#     file = request.files['file']
    
#     try:
#         # Perform object detection on the image
#         image_data = file.read()
#         image = Image.open(io.BytesIO(image_data))
        
#         # Assume `yolo_model` is the YOLO model loaded previously
#         results = yolo_model(image)
        
#         # Initialize an empty list to store the predicted class names
#         names_array = []
#         names = yolo_model.names
        
#         # Iterate through the results
#         for r in results:
#             for box in r.boxes:
#                 if box.conf <= 0.4:
#                     continue  # Skip boxes with confidence less than 0.4
#                 names_array.append(names[int(box.cls)])
        
#         # Check if any names were added
#         if not names_array:
#             return jsonify({'predictions': ''})
        
#         # Return the predictions as JSON response
#         print("This is array", names_array)

#         return jsonify({'predictions': names_array})
    
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500



@app.route('/predict', methods=['POST'])
def predict():
    # Check if the request contains a file
    file = request.files['file']
    # Perform object detection on the image
    image_data = file.read()
    image = Image.open(io.BytesIO(image_data))

    results = yolo_model(image)

    # Initialize an empty list to store the predicted class names
    class_counts = {}

    names = yolo_model.names
    # Iterate through the results
# Step 7: Iterate through the results and filter by confidence threshold
    for r in results:
        for box in r.boxes:
            cls = box.cls
            conf = box.conf
            
            # Add class to dictionary if confidence is >= 0.5
            if conf >= 0.35:
                class_name = names[int(cls)]
                if class_name in class_counts:
                    class_counts[class_name] += 1
                else:
                    class_counts[class_name] = 1

    # Step 8: Format the output based on class counts
    formatted_output = []
    for cls, count in class_counts.items():
        if count > 1:
            formatted_output.append(f"{count}{cls}")
        else:
            formatted_output.append(cls)
    # Return the predictions as JSON response
    print("Predicted class names:", formatted_output)

    return jsonify({'predictions': formatted_output})

# Load face encodings from file
def load_face_encodings():
    if os.path.exists(FACE_DATA_FILE):
        with open(FACE_DATA_FILE, "rb") as f:
            return pickle.load(f)
    return {"encodings": [], "names": []}

# Save new face encodings
def save_face_encodings(face_data):
    with open(FACE_DATA_FILE, "wb") as f:
        pickle.dump(face_data, f)

# Load stored faces
face_data = load_face_encodings()


# Path to store training images
TRAINING_DIR = "training_faces"
if not os.path.exists(TRAINING_DIR):
    os.makedirs(TRAINING_DIR)

# Path to store trained face model
MODEL_PATH = "face_model.yml"
LABELS_PATH = "labels.pkl"

# Load Haarcascade for face detection
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# LBPH Face Recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load labels
labels = {}

if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, "rb") as f:
        labels = pickle.load(f)

# Train the model if images exist
def train_model():
    images, face_labels = [], []
    label_id = 0
    for name in os.listdir(TRAINING_DIR):
        person_dir = os.path.join(TRAINING_DIR, name)
        if os.path.isdir(person_dir):
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                faces = FACE_CASCADE.detectMultiScale(img, 1.3, 5)
                for (x, y, w, h) in faces:
                    images.append(img[y:y + h, x:x + w])
                    face_labels.append(label_id)
            labels[label_id] = name
            label_id += 1

    if images:
        face_recognizer.train(images, np.array(face_labels))
        face_recognizer.save(MODEL_PATH)

        with open(LABELS_PATH, "wb") as f:
            pickle.dump(labels, f)

# Train the model initially
if os.path.exists(MODEL_PATH):
    face_recognizer.read(MODEL_PATH)
else:
    train_model()

@app.route('/add_face', methods=['POST'])
def add_face():
    # Check if the request contains a file
    file = request.files['file']# Check if the request contains a file
     # Get the name field from FormData
    name = request.form['name']   # Get the name of the person

    print("name - ",name)
    # Save the uploaded image
    image_data = file.read()
    image_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)

    # Detect face in the image
    faces = FACE_CASCADE.detectMultiScale(image, 1.3, 5)
    if faces is None or faces.size == 0:  # ✅ This correctly checks if no faces are found:
        return {"status": "error", "message": "No face detected"}

    # Save detected face in person's folder
    person_dir = os.path.join(TRAINING_DIR, name)
    if not os.path.exists(person_dir):
        os.makedirs(person_dir)

    image_path = os.path.join(person_dir, f"{len(os.listdir(person_dir)) + 1}.jpg")
    cv2.imwrite(image_path, image)

    # Retrain model
    train_model()

    return {"status": "success", "message": f"Face added for {name}"}

@app.route('/recognize_face', methods=['POST'])
def recognize_face():
    # Read image
    file = file = request.files['file']
    image_data = file.read()
    image_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)

    # Detect faces
    faces = FACE_CASCADE.detectMultiScale(image, 1.3, 5)
    if faces is None or len(faces) == 0:
        return {"status": "error", "message": "No face detected"}

    recognized_faces = []
    for (x, y, w, h) in faces:
        face = image[y:y + h, x:x + w]
        label, confidence = face_recognizer.predict(face)

        name = labels.get(label, "Unknown") if confidence < 100 else "Unknown"
        recognized_faces.append(name)

    return {"status": "success", "recognized_faces": recognized_faces}
@app.route('/', methods=['GET'])
def successget():
    print("hello world")
    return "Hello World"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


# from flask import Flask, request, jsonify
# from inference_sdk import InferenceHTTPClient
# from PIL import Image
# import io
# from ultralytics import YOLO
# from flask import jsonify

# app = Flask(__name__)

# CLIENT = InferenceHTTPClient(
#     api_url="https://classify.roboflow.com",
#     api_key="y5O5K3Tvyj7jS6zhNhAr"
# )
# DollarCLIENT = InferenceHTTPClient(
#     api_url="https://detect.roboflow.com",
#     api_key="y5O5K3Tvyj7jS6zhNhAr"
# )


# # Load the YOLO model
# model = YOLO("yolov8n.pt")  # Load a pretrained model

# @app.route('/hello', methods=['GET'])
# def hello():
#     return 'Hello, World!'


# @app.route('/currencyprediction', methods=['POST'])
# def currencyprediction():
#     # Check if the request contains a file
#     file = request.files['file']
    
#     # Convert image to bytes
#     image_data = file.read()
#     image = Image.open(io.BytesIO(image_data))

#     # Perform inference using the PKR model
    
#     usresult = DollarCLIENT.infer(image, model_id="usd-money/2")
#     classes = [prediction['class'] for prediction in usresult['predictions']]
#     confidences = [float(prediction['confidence']) for prediction in usresult['predictions']]
#     any_greater_than_50 = any(confidence > 50 for confidence in confidences)
#     print(classes)
#     print(any_greater_than_50)
#     print("confidence")
#     print(confidences)
#     if confidences:  # Check if confidences is not empty
#      print(confidences[0])
#     # Check if the predicted classes are not "other"
#     if confidences and confidences[0]>0.5:
#         print("inside")
#         print(classes)

#         return jsonify({'classes': classes})
#     else:
#         # Perform inference using the USD model
#         pkrresult = CLIENT.infer(image, model_id="pcs-d_app/1")
#         predicted_classes = pkrresult.get('predicted_classes', [])
#         print(predicted_classes)
#         # Return the classes from the USD model
#         return jsonify({'classes': predicted_classes})


# @app.route('/predict', methods=['POST'])
# def predict():
#     # Check if the request contains a file
#     file = request.files['file']
#     # Perform object detection on the image
#     image_data = file.read()
#     image = Image.open(io.BytesIO(image_data))

#     results = model(image)

#     # Initialize an empty list to store the predicted class names
#     names_array = []
#     names = model.names
#     # Iterate through the results
#     for r in results:
#         for c in r.boxes.cls:
#             # Append the value of names[int(c)] to the names_array
#             names_array.append(names[int(c)])

#     # Return the predictions as JSON response
#     return jsonify({'predictions': names_array})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)
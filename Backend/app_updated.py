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
import face_recognition
from datetime import datetime
from deepface import DeepFace
import json
from mtcnn import MTCNN
from scipy.spatial.distance import cosine
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

app = Flask(__name__)

# Load the ResNet50V2 model
model_path = 'ResNet50V2.h5'  # Update with your model path
resnet_model = load_model(model_path)
# File to store known faces
FACE_DATA_FILE = "face_encodings.pkl"


# Constants
DB_PATH = "face_db.json"
MODEL_NAME = "Facenet512"  # Change to ArcFace, VGG-Face, etc., if needed
IMAGE_STORAGE_DIR = "stored_faces"
# Load BLIP model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Ensure the storage directory exists
if not os.path.exists(IMAGE_STORAGE_DIR):
    os.makedirs(IMAGE_STORAGE_DIR)

# Initialize MTCNN detector
detector = MTCNN()

# Define the class names for ResNet50V2
class_names = [
    '1000Rs', '1000Rsback', '100Rs', '100Rsback', '10Rs', '10Rsback',
    '20Rs', '20Rsback', '5000Rs', '5000Rsback', '500Rs', '500Rsback',
    '50Rs', '50Rsback'
]

# Load the YOLO model
# yolo_model = YOLO("yolov8x-oiv7.pt")  # Load a pretrained model
yolo_model = YOLO("yolov8n.pt")  # Load a pretrained model

# verification = DeepFace.verify(img1_path = "training_faces/obama-28736487236.jpg", img2_path = "training_faces/adil-20250324-002619.jpg")
# print("Verification: ", verification)

def generate_caption(image):
    image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    output = model.generate(**inputs)
    return processor.decode(output[0], skip_special_tokens=True)

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
    caption = generate_caption(image)
    print("Caption: ", caption)

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

    return jsonify({'predictions': formatted_output, "caption": caption})

# --- FACE CROPPING FUNCTION ---
def crop_face(image_path):
    """Detects and crops the face from an image."""
    print(f"Reading image from {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print("Failed to read image")
        return None

    # Resize the image to a smaller size to reduce memory usage
    img = cv2.resize(img, (640, 480))

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img_rgb)

    if faces:
        print(f"Detected faces: {faces}")
        x, y, width, height = faces[0]['box']
        cropped_face = img_rgb[y:y+height, x:x+width]
        return Image.fromarray(cropped_face)
    else:
        print("No faces detected")
        return None

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256))  # Resize to standard size
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img = cv2.equalizeHist(img)  # Improve contrast
    return img

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
# face_recognizer = cv2.face.LBPHFaceRecognizer_create()

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
        # face_recognizer.train(images, np.array(face_labels))
        # face_recognizer.save(MODEL_PATH)

        with open(LABELS_PATH, "wb") as f:
            pickle.dump(labels, f)

# Train the model initially
if os.path.exists(MODEL_PATH):
    print('hey')
    # face_recognizer.read(MODEL_PATH)
else:
    train_model()


FACE_DB_PATH = "face_db.json"  # Store embeddings

# Load existing faces
def load_face_db():
    if os.path.exists(FACE_DB_PATH):
        with open(FACE_DB_PATH, "r") as f:
            return json.load(f)
    return {}

# Save new faces
def save_face_db(data):
    with open(FACE_DB_PATH, "w") as f:
        json.dump(data, f)

face_db = load_face_db()

def add_predefined_face(path, name):
    face = crop_face(path)
    if face is None:
        print("No face detected. Try another image.")
        return {"status": "error", "message": "No face detected. Try another image."}, 400
    # Ensure the storage directory exists
    if not os.path.exists(IMAGE_STORAGE_DIR):
        os.makedirs(IMAGE_STORAGE_DIR)
    # Save cropped face for reference
    stored_image_path = os.path.join(IMAGE_STORAGE_DIR, f"{name}.jpg")
    face.save(stored_image_path)

    # Generate encoding
    embedding = DeepFace.represent(img_path=stored_image_path, model_name=MODEL_NAME, enforce_detection=False)[0]["embedding"]

    # Load or initialize database
    try:
        with open(DB_PATH, "r") as f:
            face_db = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        face_db = {}

    # Store multiple encodings per person
    if name in face_db:
        face_db[name].append(embedding)
    else:
        face_db[name] = [embedding]

    # Save back to database
    with open(DB_PATH, "w") as f:
        json.dump(face_db, f)

    print(f"Face added successfully for {name}.")
    return {"status": "success", "message": "Photo added and face encodings updated successfully"}

@app.route('/add_face', methods=['POST'])
def add_face():
    global face_db
    print("Request files:", request.files)
    print("Request form:", request.form)
    
    # Check if the request contains a file
    if 'file' not in request.files:
        print("No file part in the request")
        return {"status": "error", "message": "No file part in the request"}, 400

    file = request.files['file']  # Get the file from the request
    name = request.form.get('name', 'Unknown')  # Get the name of the person

    print("name - ", name)

    # Save the uploaded image
    file.save("new_image.jpg")
    face = crop_face("new_image.jpg")
    if face is None:
        print("No face detected. Try another image.")
        return {"status": "error", "message": "No face detected. Try another image."}, 400
    # Ensure the storage directory exists
    if not os.path.exists(IMAGE_STORAGE_DIR):
        os.makedirs(IMAGE_STORAGE_DIR)
    # Save cropped face for reference
    stored_image_path = os.path.join(IMAGE_STORAGE_DIR, f"{name}.jpg")
    face.save(stored_image_path)

    # Generate encoding
    embedding = DeepFace.represent(img_path=stored_image_path, model_name=MODEL_NAME, enforce_detection=False)[0]["embedding"]

    # Load or initialize database
    try:
        with open(DB_PATH, "r") as f:
            face_db = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        face_db = {}

    # Store multiple encodings per person
    if name in face_db:
        face_db[name].append(embedding)
    else:
        face_db[name] = [embedding]

    # Save back to database
    with open(DB_PATH, "w") as f:
        json.dump(face_db, f)

    print(f"Face added successfully for {name}.")
    return {"status": "success", "message": "Photo added and face encodings updated successfully"}

@app.route('/recognize_face', methods=['POST'])
def recognize_face():
    global face_db
    file = request.files['file']
    img_path = "temp_recognition.jpg"
    file.save(img_path)

    face = crop_face(img_path)
    if face is None:
        print("No face detected in the image.")
        return {"status": "error", "message": "No face detected."}, 400
    
      # Save temporary cropped face
    temp_image_path = "temp_cropped.jpg"
    face.save(temp_image_path)

    # Generate encoding for the input image
    input_embedding = DeepFace.represent(img_path=temp_image_path, model_name=MODEL_NAME, enforce_detection=False)[0]["embedding"]
    input_embedding = np.array(input_embedding).flatten()


    # Load stored encodings
    try:
        with open(DB_PATH, "r") as f:
            face_db = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print("No stored faces found.")
        return {"status": "error", "message": "No stored faces"}, 400

    # Compare with stored embeddings
    best_match = None
    min_distance = float("inf")

    for name, encodings in face_db.items():
        for stored_embedding in encodings:
            stored_embedding = np.array(stored_embedding).flatten()  # Ensure it's 1D
            if stored_embedding.shape != input_embedding.shape:
                print(f"⚠️ Shape mismatch for {name}: {stored_embedding.shape} vs {input_embedding.shape}")
                continue  
            distance = cosine(input_embedding, stored_embedding)
            if distance < 0.3 and distance < min_distance:
                min_distance = distance
                best_match = name
    print("best_match", best_match)
    if best_match:
        return {"status": "success", "message": f"Recognized as {best_match}"}
    else:
        return {"status": "error", "message": "Face not recognized"}

    # try:
    #     embedding = DeepFace.represent(img_path, model_name="Facenet")[0]["embedding"]
    #     os.remove(img_path)

    #     best_match = None
    #     min_distance = float("inf")

    #     for name, stored_embedding in face_db.items():
    #         distance = np.linalg.norm(np.array(stored_embedding) - np.array(embedding))
    #         if distance < min_distance:
    #             min_distance = distance
    #             best_match = name

    #     if best_match and min_distance < 0.6:  # Adjust threshold as needed
    #         return jsonify({"message": f"Recognized as {best_match}"}), 200
    #     else:
    #         print('kiteela??')
    #         return jsonify({"message": "Face not recognized"}), 400

    # except Exception as e:
    #     return jsonify({"error": str(e)}), 400
            
                
    #             # If face encodings are found, add them to the array
    #             if face_encodings:
    #                 known_face_encodings.append(face_encodings[0])
    #                 known_face_names.append(filename.split('-')[0])
    #     if not known_face_encodings:
    #         print("No known faces found ")
    #         return {"status": "error", "message": "No known faces found"}, 400
    #     else:
    #         # Save the updated face encodings and names to a file
    #         face_data = {"encodings": known_face_encodings, "names": known_face_names}
    #         with open(FACE_DATA_FILE, "wb") as f:
    #             pickle.dump(face_data, f)

    # # Load new image and encode
    # print("file_path", file_path)
    # new_image = preprocess_image(file_path)
    # # rgb_new_img = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    # new_encodings = face_recognition.face_encodings(new_image)

    # if not new_encodings:
    #     print("No faces found in the image")
    #     return {"status": "error", "message": "No faces found in the image"}, 400

    # new_encoding = new_encodings[0]

    # # Compare against all stored faces
    # for known_encoding, name in zip(known_face_encodings, known_face_names):
    #     match = face_recognition.compare_faces([known_encoding], new_encoding)
    #     if match[0]:
    #         print(f"✅ Recognized as {name}!")
    #         return {"status": "success", "recognized_faces": name}

    # print("❌ No match found.")
    # return {"status": "error", "message": "No match found."}








    # # Load stored face encodings
    # if os.path.exists("saved_faces.pkl"):
    #     with open("saved_faces.pkl", "rb") as file:
    #         known_faces = pickle.load(file)
    # else:
    #     return {"status": "error", "message": "No known faces found"}, 400

    # # Load new image and encode
    # new_image = face_recognition.load_image_file(file_path)
    # new_encodings = face_recognition.face_encodings(new_image)

    # if not new_encodings:
    #     print("No faces found in the image")
    #     return {"status": "error", "message": "No faces found in the image"}, 400

    # new_encoding = new_encodings[0]

    # # Compare against all stored faces
    # for name, saved_encoding in known_faces.items():
    #     match = face_recognition.compare_faces([saved_encoding], new_encoding)
    #     if match[0]:
    #         print(f"✅ Recognized as {name}!")
    #         return {"status": "success", "recognized_faces": name}

    # print("❌ No match found.")
    # return {"status": "error", "message": "No match found."}

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
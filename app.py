from flask import Flask, render_template, Request, Response, jsonify
import cv2
import numpy as np
import os
from datetime import datetime
import pytz
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Pastikan struktur folder benar
required_folders = ['dataset_wajah', 'models', 'templates']
for folder in required_folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

def load_models():
    # Load and preprocess dataset first
    print("Loading dataset...")
    dataset_path = './dataset_wajah/'
    faces = []
    labels = []
    
    for label in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, label)
        if os.path.isdir(person_path):
            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img = load_img(img_path, color_mode='grayscale', target_size=(64, 64))
                    img_array = img_to_array(img)
                    faces.append(img_array)
                    labels.append(label)

    faces = np.array(faces) / 255.0
    
    # Convert labels
    unique_labels = list(set(labels))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    labels = np.array([label_map[label] for label in labels])
    labels = to_categorical(labels, num_classes=len(unique_labels))
    
    # Split data
    x_train, x_test, y_train, y_test = train_test_split(faces, labels, test_size=0.2, random_state=42)

    # Try to load or create CNN model
    try:
        print("Trying to load existing CNN model...")
        cnn_model = load_model('./models/model_cnn.h5')
    except:
        print("CNN model not found, creating new CNN model...")
        cnn_model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(len(unique_labels), activation='softmax')
        ])
        
        print("Training CNN model...")
        cnn_model.compile(optimizer='adam',
                         loss='categorical_crossentropy',
                         metrics=['accuracy'])
        cnn_model.fit(x_train, y_train,
                     epochs=5,
                     batch_size=32,
                     validation_data=(x_test, y_test),
                     verbose=1)
        cnn_model.save('./models/model_cnn.h5')
    
    # Create DNN models
    print("Creating DNN models...")
    
    # DNN Model 1 - Deep and Wide
    dnn1 = Sequential([
        Flatten(input_shape=(64, 64, 1)),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(len(unique_labels), activation='softmax')
    ])
    
    # DNN Model 2 - Shallow but Wide
    dnn2 = Sequential([
        Flatten(input_shape=(64, 64, 1)),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(len(unique_labels), activation='softmax')
    ])
    
    # DNN Model 3 - Deep but Narrow
    dnn3 = Sequential([
        Flatten(input_shape=(64, 64, 1)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(len(unique_labels), activation='softmax')
    ])
    
    # Compile and train models
    print("Training DNN models...")
    for i, model in enumerate([dnn1, dnn2, dnn3], 1):
        print(f"Training DNN model {i}...")
        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        model.fit(x_train, y_train,
                 epochs=5,
                 batch_size=32,
                 validation_data=(x_test, y_test),
                 verbose=1)
        model.save(f'./models/dnn_model{i}.h5')
    
    print("All models loaded/created successfully!")
    
    return {
        'dnn1': dnn1,
        'dnn2': dnn2,
        'dnn3': dnn3,
        'cnn': cnn_model
    }

# Label mapping
label_map = {0: "Jonathan", 1: "Yessa", 2: "Teman3"}

# Inisialisasi models
models = load_models()

def predict_face(frame, models):
    # Preprocess frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(gray, (64, 64))
    face = img_to_array(face) / 255.0
    face = np.expand_dims(face, axis=0)

    # Get predictions from all models
    predictions = {}
    for model_name, model in models.items():
        pred = model.predict(face)
        predictions[model_name] = label_map[np.argmax(pred)]

    # Voting system
    final_prediction = max(set(predictions.values()), key=list(predictions.values()).count)
    confidence_scores = {name: pred for name, pred in predictions.items()}
    
    return final_prediction, confidence_scores

def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Get prediction
        name, scores = predict_face(frame, models)
        
        # Add text to frame
        cv2.putText(frame, f'Name: {name}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Convert frame to jpg
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def save_attendance(name):
    wib_tz = pytz.timezone('Asia/Jakarta')
    current_time_wib = datetime.now(wib_tz).strftime('%Y-%m-%d %H:%M:%S')
    
    attendance_data = {
        'Nama': [name],
        'Tanggal': [current_time_wib.split()[0]],
        'Waktu': [current_time_wib.split()[1]],
        'Status': ['Hadir']
    }
    
    df = pd.DataFrame(attendance_data)
    csv_path = 'attendance.csv'
    
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        df = pd.concat([existing_df, df], ignore_index=True)
    
    df.to_csv(csv_path, index=False)
    return current_time_wib

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture():
    camera = cv2.VideoCapture(0)
    success, frame = camera.read()
    camera.release()
    
    if success:
        name, scores = predict_face(frame, models)
        timestamp = save_attendance(name)
        
        return jsonify({
            'name': name,
            'timestamp': timestamp,
            'model_predictions': scores
        })
    
    return jsonify({'error': 'Failed to capture image'})

@app.route('/attendance')
def attendance():
    if os.path.exists('attendance.csv'):
        df = pd.read_csv('attendance.csv')
        attendance_data = df.to_dict('records')
    else:
        attendance_data = []
    return render_template('attendance.html', attendance=attendance_data)

if __name__ == '__main__':
    app.run(debug=True) 
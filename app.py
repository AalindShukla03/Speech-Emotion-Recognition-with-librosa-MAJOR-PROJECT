from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
import librosa
import numpy as np
from keras.models import load_model

app = Flask(__name__)

# Constants for the model and file upload
import os

# Print the current working directory
print("Current Working Directory:", os.getcwd())

# Change the current working directory
new_directory = r'D:\MAJOR PROJECT\major project complete'
os.chdir(new_directory)

# Print the updated current working directory
print("Updated Working Directory:", os.getcwd())

MODEL_PATH = "model\\hello.h5"
app.config['STATIC_FOLDER'] = 'static'
app.config['UPLOAD_FOLDER'] = 'uploads'
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav'}

EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised']

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the pre-trained model
model = load_model(MODEL_PATH)

# ... (other functions and routes)

# Function to predict emotion from the features
def predict_emotion(features):
    # Predict using the pre-trained model
    prediction = model.predict(features.reshape(1, -1))
    
    # Get the index of the emotion with the highest probability
    predicted_class_index = np.argmax(prediction)

    # Get the name of the predicted emotion
    predicted_emotion_name = EMOTIONS[predicted_class_index]

    return predicted_emotion_name

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the pre-trained model
model = load_model(MODEL_PATH)

# Function to check if the file has a valid extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to extract features from an audio file
def extract_features(file_path):
    X, sample_rate = librosa.load(file_path)
    # Extracting features (Placeholder, add actual code)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=librosa.stft(X), sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=librosa.stft(X), sr=sample_rate).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
    return np.hstack([mfccs, chroma, mel, contrast, tonnetz])

# Function to predict emotion from the features
# Function to predict emotion from the features
def predict_emotion(features):
    # Predict using the pre-trained model
    prediction = model.predict(features.reshape(1, -1))
    
    # Get the index of the emotion with the highest probability
    predicted_class_index = np.argmax(prediction)

    # Get the name of the predicted emotion
    predicted_emotion_name = EMOTIONS[predicted_class_index]

    print("Predicted Emotion:", predicted_emotion_name)  # Add this line

    return predicted_emotion_name

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle the file upload
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='No selected file')

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        file.save(file_path)

        # Extract features
        features = extract_features(file_path)

        # Predict emotion
        predicted_emotion = predict_emotion(features)

        return render_template('result.html', filename=filename, emotion=predicted_emotion)

    return render_template('index.html', error='Invalid file format')

if __name__ == '__main__':
    app.run(debug=True)

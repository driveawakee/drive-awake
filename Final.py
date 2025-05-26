import cv2
import numpy as np
import time
import pygame
import os
import wave
import struct
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
# matplotlib removed to avoid import errors

# Initialize pygame for alarm sound
pygame.init()

# Create a folder for the alarm sound if it doesn't exist
if not os.path.exists('sounds'):
    os.makedirs('sounds')

# Path for the alarm sound
alarm_sound_path = 'alarm.wav'

# Create a simple beep sound if alarm.wav doesn't exist
if not os.path.exists(alarm_sound_path):
    print("Creating a default alarm sound...")
    
    # Parameters for the sound
    frequency = 2500  # Hz
    duration = 1.0  # seconds
    sample_rate = 44100  # Hz
    amplitude = 32767  # Maximum amplitude for 16-bit audio
    
    # Generate a simple sine wave
    num_samples = int(duration * sample_rate)
    samples = []
    
    for i in range(num_samples):
        t = float(i) / sample_rate
        sample = amplitude * np.sin(2 * np.pi * frequency * t)
        samples.append(int(sample))
    
    # Write the samples to a WAV file
    with wave.open(alarm_sound_path, 'w') as wav_file:
        # 1 channel, 2 bytes per sample, sample_rate samples per second
        wav_file.setparams((1, 2, sample_rate, num_samples, 'NONE', 'not compressed'))
        
        # Convert the samples to binary data
        binary_samples = struct.pack('h' * len(samples), *samples)
        wav_file.writeframes(binary_samples)
    
    print(f"Alarm sound created at {alarm_sound_path}")

# Load Haar cascade for face detection
# We'll still use this for face detection, then use CNN for eye state detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Constants
EYE_AR_CONSEC_FRAMES = 10  # Reduced from 15 as CNN can be more accurate
COUNTER = 0  # Frame counter
ALARM_ON = False  # Alarm status
MODEL_PATH = 'drowsiness_model.h5'
EYE_SIZE = (24, 24)  # Size of eye images for CNN
CONFIDENCE_THRESHOLD = 0.75  # Confidence threshold for "closed" prediction

class EyeStateModel:
    def __init__(self):
        self.model = None
        
        
    def build_model(self):
        """Build the CNN model architecture"""
        model = Sequential()
        
        # First Convolutional Layer
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(EYE_SIZE[0], EYE_SIZE[1], 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        # Second Convolutional Layer
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        # Third Convolutional Layer
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        # Flatten and Dense Layers
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))  # Prevents overfitting
        model.add(Dense(2, activation='softmax'))  # 2 classes: open, closed
        
        # Compile model
        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(learning_rate=0.001),
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
        """Train the CNN model with data"""
        if self.model is None:
            self.build_model()
        
        # Data augmentation to improve generalization
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2]
        )
        
        # Train the model
        history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            epochs=epochs,
            validation_data=(X_val, y_val),
            verbose=1
        )
        
        return history
    
    def save_model(self, path=MODEL_PATH):
        """Save the trained model"""
        if self.model is not None:
            self.model.save(path)
            print(f"Model saved to {path}")
        else:
            print("No model to save")
    
    def load_model(self, path=MODEL_PATH):
        """Load a trained model"""
        try:
            self.model = load_model(path)
            print(f"Model loaded from {path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict(self, eye_image):
        """Predict whether an eye is open or closed"""
        if self.model is None:
            print("Model not loaded")
            return None
        
        # Preprocess the image
        eye_image = cv2.resize(eye_image, (64,64))
        eye_image = eye_image / 255.0  # Normalize
        eye_image = np.expand_dims(eye_image, axis=0)  # Add batch dimension
        eye_image = np.expand_dims(eye_image, axis=-1)  # Add channel dimension for grayscale
        
        # Make prediction
        prediction = self.model.predict(eye_image, verbose=0)
        return prediction[0]

def extract_eyes(frame, face):
    """Extract eye regions from a face"""
    (x, y, w, h) = face
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Define eye regions based on facial proportions
    # Left eye is typically in the upper-left quadrant of the face
    left_x = int(x + w * 0.1)
    left_y = int(y + h * 0.25)
    left_w = int(w * 0.4)
    left_h = int(h * 0.2)
    
    # Right eye is typically in the upper-right quadrant of the face
    right_x = int(x + w * 0.5)
    right_y = int(y + h * 0.25)
    right_w = int(w * 0.4)
    right_h = int(h * 0.2)
    
    # Extract eye regions
    left_eye = gray[left_y:left_y+left_h, left_x:left_x+left_w]
    right_eye = gray[right_y:right_y+right_h, right_x:right_x+right_w]
    
    # Draw rectangles around eyes for visualization
    cv2.rectangle(frame, (left_x, left_y), (left_x+left_w, left_y+left_h), (0, 255, 0), 2)
    cv2.rectangle(frame, (right_x, right_y), (right_x+right_w, right_y+right_h), (0, 255, 0), 2)
    
    return left_eye, right_eye, frame

def detect_drowsiness(frame, model):
    """
    Detect faces and eyes in the frame and determine drowsiness
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # If no faces detected, return frame with no detected eyes
    if len(faces) == 0:
        return frame, True  # No face detected, consider as "eyes closed" for safety
    
    # Process the first face found
    face = faces[0]
    cv2.rectangle(frame, (face[0], face[1]), (face[0]+face[2], face[1]+face[3]), (255, 0, 0), 2)
    
    # Extract eyes
    left_eye, right_eye, frame = extract_eyes(frame, face)
    
    # Check if eye regions are valid
    if left_eye.size == 0 or right_eye.size == 0:
        return frame, True  # Invalid eye regions, consider as "eyes closed" for safety
    
    # Predict eye states
    left_pred = model.predict(left_eye)
    right_pred = model.predict(right_eye)
    
    # Get probabilities of closed eyes (index 1 is for "closed" class)
    left_closed_prob = float(left_pred)
    right_closed_prob = float(right_pred)
    
    # Average probability of eyes being closed
    avg_closed_prob = (left_closed_prob + right_closed_prob) / 2
    
    # Display probabilities on frame
    cv2.putText(frame, f"Left eye closed: {left_closed_prob:.2f}", (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, f"Right eye closed: {right_closed_prob:.2f}", (10, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Determine if eyes are closed based on threshold
    eyes_closed = avg_closed_prob > CONFIDENCE_THRESHOLD
    
    return frame, eyes_closed

def calculate_drowsiness(eyes_closed):
    """
    Calculate drowsiness based on consecutive frames with closed eyes
    """
    global COUNTER
    
    if eyes_closed:
        COUNTER += 1
        # If eyes closed for a sufficient number of frames, trigger alarm
        if COUNTER >= EYE_AR_CONSEC_FRAMES:
            return True
    else:
        # Reset the counter if eyes are open
        COUNTER = 0
        
    return False

def collect_training_data():
    """
    Helper function to collect training data.
    This would be used to create a dataset for training the CNN.
    """
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Create directories for saving images
    if not os.path.exists('dataset'):
        os.makedirs('dataset/open')
        os.makedirs('dataset/closed')
    
    print("Press 'o' to capture open eyes, 'c' to capture closed eyes, 'q' to quit")
    
    open_count = 0
    closed_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Display frame
        cv2.imshow("Data Collection", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('o'):  # Capture open eyes
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                face = faces[0]
                left_eye, right_eye, _ = extract_eyes(frame, face)
                
                if left_eye.size > 0 and right_eye.size > 0:
                    # Save left eye
                    left_eye_resized = cv2.resize(left_eye, EYE_SIZE)
                    cv2.imwrite(f'dataset/open/left_{open_count}.jpg', left_eye_resized)
                    
                    # Save right eye
                    right_eye_resized = cv2.resize(right_eye, EYE_SIZE)
                    cv2.imwrite(f'dataset/open/right_{open_count}.jpg', right_eye_resized)
                    
                    open_count += 1
                    print(f"Captured open eyes: {open_count}")
        
        elif key == ord('c'):  # Capture closed eyes
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                face = faces[0]
                left_eye, right_eye, _ = extract_eyes(frame, face)
                
                if left_eye.size > 0 and right_eye.size > 0:
                    # Save left eye
                    left_eye_resized = cv2.resize(left_eye, EYE_SIZE)
                    cv2.imwrite(f'dataset/closed/left_{closed_count}.jpg', left_eye_resized)
                    
                    # Save right eye
                    right_eye_resized = cv2.resize(right_eye, EYE_SIZE)
                    cv2.imwrite(f'dataset/closed/right_{closed_count}.jpg', right_eye_resized)
                    
                    closed_count += 1
                    print(f"Captured closed eyes: {closed_count}")
        
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Data collection complete: {open_count} open eyes, {closed_count} closed eyes")

def prepare_dataset():
    """
    Prepare the dataset for training from collected images
    """
    if not os.path.exists('dataset/open') or not os.path.exists('dataset/closed'):
        print("Dataset not found. Please collect data first.")
        return None, None
    
    # Lists to store data and labels
    X = []
    y = []
    
    # Load open eyes images
    for img_name in os.listdir('dataset/open'):
        if img_name.endswith('.jpg'):
            img_path = os.path.join('dataset/open', img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, EYE_SIZE)
                X.append(img)
                y.append(0)  # 0 for open eyes
    
    # Load closed eyes images
    for img_name in os.listdir('dataset/closed'):
        if img_name.endswith('.jpg'):
            img_path = os.path.join('dataset/closed', img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, EYE_SIZE)
                X.append(img)
                y.append(1)  # 1 for closed eyes
    
    if len(X) == 0:
        print("No images found in dataset.")
        return None, None
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Normalize images
    X = X / 255.0
    
    # Reshape for CNN (add channel dimension)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    
    # Convert labels to one-hot encoding
    y = tf.keras.utils.to_categorical(y, 2)
    
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Dataset prepared: {len(X_train)} training samples, {len(X_val)} validation samples")
    return (X_train, y_train), (X_val, y_val)

def train_cnn_model():
    """
    Train the CNN model using the prepared dataset
    """
    # Prepare the dataset
    (X_train, y_train), (X_val, y_val) = prepare_dataset()
    
    if X_train is None:
        print("Failed to prepare dataset.")
        return False
    
    # Create and train the model
    model = EyeStateModel()
    model.build_model()
    
    history = model.train_model(X_train, y_train, X_val, y_val, epochs=15)
    
    # Print training history instead of plotting
    print("Training History Summary:")
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Final training loss: {history.history['loss'][-1]:.4f}")
    print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
    
    # Save the model
    model.save_model()
    
    print("Model training complete!")
    return True

def main():
    global COUNTER, ALARM_ON
    
    # Check if model exists, otherwise offer to train or use the dummy model
    model = EyeStateModel()
    model_loaded = model.load_model()
    model.summary = model.model.summary() if model.model else None
    if not model_loaded:
        print("No trained model found. You have the following options:")
        print("1. Collect training data and train a new model")
        print("2. Use a pre-trained model (not available)")
        print("3. Continue with a basic model (may not be accurate)")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            print("Starting data collection...")
            collect_training_data()
            print("Training model...")
            train_cnn_model()
            model_loaded = model.load_model()
        elif choice == '2':
            print("Pre-trained model option selected, but no pre-trained model is available.")
            print("Building a basic model instead...")
            model.build_model()
        else:
            print("Building a basic model...")
            model.build_model()
    
    # Start video capture from webcam
    cap = cv2.VideoCapture(0)
    
    # Set the frame dimensions
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    print("Drowsiness detection started. Press 'q' to quit.")
    
    # Load the sound for alarm
    pygame.mixer.init()
    try:
        sound = pygame.mixer.Sound(alarm_sound_path)
    except Exception as e:
        print(f"Error loading sound file: {e}")
        print("Continuing without sound.")
        sound = None
    
    while True:
        # Read a frame from the video capture
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to grab frame.")
            break
        
        # Detect drowsiness using the CNN model
        frame, eyes_closed = detect_drowsiness(frame, model)
        
        # Calculate drowsiness
        is_drowsy = calculate_drowsiness(eyes_closed)
        
        # Handle drowsiness alert
        if is_drowsy:
            if not ALARM_ON:
                ALARM_ON = True
                if sound:
                    sound.play(-1)  # Loop the alarm sound
                
            # Display warning message
            cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # Stop the alarm if no longer drowsy
            if ALARM_ON:
                ALARM_ON = False
                if sound:
                    sound.stop()
        
        # Display status
        status = "Eyes Closed" if eyes_closed else "Eyes Open"
        cv2.putText(frame, f"Status: {status}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display counter
        cv2.putText(frame, f"Counter: {COUNTER}/{EYE_AR_CONSEC_FRAMES}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display the current frame
        cv2.imshow("Drowsiness Detection (CNN)", frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    if sound and ALARM_ON:
        sound.stop()
    pygame.quit()

if __name__ == "__main__":
    main()
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the pre-trained emotion detection model
model = load_model('fer2013_mini_XCEPTION.110-0.65.hdf5')

# Emotion labels as per the FER-2013 dataset
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# OpenCV's pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam capture
cap = cv2.VideoCapture(0)

# Skip frames for optimization (process every nth frame)
frame_skip = 5
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Skip frames to improve real-time processing speed
    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    # Convert frame to grayscale (the model expects grayscale images)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Crop and resize the detected face to (64, 64), as required by the model
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (64, 64))  # Resize to (64, 64) as expected by the model
        face_array = np.expand_dims(face_resized, axis=0)
        face_array = np.expand_dims(face_array, axis=-1)  # Add channel dimension
        face_array = face_array / 255.0  # Normalize pixel values

        # Make emotion prediction
        emotion_probs = model.predict(face_array)
        emotion = emotion_labels[np.argmax(emotion_probs)]

        # Draw rectangle and emotion label on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)

    # Exit the webcam feed on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and close the window
cap.release()
cv2.destroyAllWindows()

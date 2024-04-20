'''import cv2

# Load one of the collected images
image_path = "Data/1/1_1.jpg"  # Replace with the path to your image
img = cv2.imread(image_path)

# Get the frame size
frame_height, frame_width, _ = img.shape

print(f"Frame size of the image: {frame_width}x{frame_height}")

import cv2
import numpy as np
import math
import os
import mediapipe as mp

from cvzone.ClassificationModule import Classifier

cap = cv2.VideoCapture(0)

# Initialize the hand tracking and classification modules
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
classifier = Classifier("sign_language_recognition_model.h5", "labels.txt")

# Constants
imgSize = 100  # Model input image size
offset = 30  # Offset for cropping hand region
folder = "dataset"  # Folder to save captured images

# Create folder if it doesn't exist
os.makedirs(folder, exist_ok=True)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read from camera.")
        break
    
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract hand region
            min_x, min_y, max_x, max_y = float('inf'), float('inf'), 0, 0
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * img.shape[1]), int(landmark.y * img.shape[0])
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
            a = min_x - offset
            b = min_y - offset
            c = max_x + offset
            d = max_y + offset
            w = c - a
            h = d - b
            
            # Crop and resize hand region
            imgCrop = img[b:b+h, a:a+w]
            imgResize = cv2.resize(imgCrop, (imgSize, imgSize))
            imgInput = np.expand_dims(imgResize, axis=0) / 255.0  # Normalize image and add batch dimension

            # Perform gesture recognition
            prediction, _ = classifier.getPrediction(imgInput)
            
            # Display prediction on the image
            cv2.putText(img, prediction, (a, b - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Show hand region and resized image
            cv2.imshow("Hand Region", imgCrop)
            cv2.imshow("Resized Image", imgResize)

    # Display the original image with hand tracking
    cv2.imshow("Hand Tracking", img)
    
    key = cv2.waitKey(1)
    if key == ord("m"):
        # Save captured image
        cv2.imwrite(f'{folder}/{prediction}_{len(os.listdir(folder))}.jpg', imgResize)
        print("Image saved.")

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()'''

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import math
from cvzone.HandTrackingModule import HandDetector
from tensorflow.keras.models import load_model

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

imgSize = 550
offset = 30

counter = 0


# Load the trained CNN model
model = load_model("sign_language_recognition_model.h5")

# Initialize hand detector
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Define image size
img_size = 100

# Define class labels
class_labels = {0: 'A', 1: 'B', 2: 'C'}  # Replace with your actual class labels

# Open the default camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, img = cap.read()
    if not success:
        continue
    
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        min_x, min_y, max_x, max_y = float('inf'), float('inf'), 0, 0
        
        for hand_landmarks in results.multi_hand_landmarks:
            keypoints = []

            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * img.shape[1]), int(landmark.y * img.shape[0])
                keypoints.append((x, y))
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)

                a = min_x - offset
                b = min_y - offset
                c = max_x + offset
                d = max_y + offset
                w = c - a
                h = d - b

        cv2.rectangle(img, (a, b), (c, d), (0, 255, 0), 2)

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[b:b+h, a:a+w]

        cv2.imshow("ImageCrop", imgCrop)

        imgCropShape = imgCrop.shape
        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 4)
            imgWhite[:, wGap:wCal+wGap] = imgResize
            
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 4)
            imgWhite[hGap:hCal+hGap, :] = imgResize
        # Make prediction
        prediction = model.predict(imgWhite)
        predicted_class = np.argmax(prediction)
        label = class_labels[predicted_class]
        
        # Display the predicted label
        cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    # Display the image
    cv2.imshow("Gesture Recognition", img)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()


import cv2
import mediapipe as mp
import numpy as np
import math
import os

from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
#from keras.models import load_model

#model = load_model('sign_language_recognition_model.h5')
detector = HandDetector(maxHands=2)
classifier = Classifier("sign_language_recognition_model.h5","labels.txt")

img='dataset/1/1_20.jpg'
prediction, index = Classifier.getPrediction(img)
accuracy=prediction[index]
print(prediction, index)

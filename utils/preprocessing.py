import cv2
import numpy as np

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)        
    resized = cv2.resize(gray, (28, 28))                  
    normalized = resized / 255.0                          
    reshaped = normalized.reshape(1, 28, 28, 1)           
    return reshaped.astype(np.float32)
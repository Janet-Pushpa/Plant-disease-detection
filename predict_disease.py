import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

def resize_image(image, new_size=(128, 128)):
    return cv2.resize(image, new_size)

def extract_hog_features(image):
    # Convert the image to grayscale using cv2
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor()
    # Compute HOG features
    hog_features = hog.compute(gray_image)
    return hog_features.flatten()

def predict_disease(image_path):
    model = joblib.load('plant_disease_model.joblib')  # Load the saved model
    
    # Load and preprocess the image
    image = cv2.imread(image_path)
    if image is None:
        return "Error: Could not load image"
    
    resized_image = resize_image(image)
    features = extract_hog_features(resized_image)
    features = features.reshape(1, -1)  # Ensure the feature array has the right shape
    
    # Make prediction
    prediction = model.predict(features)
    
    # Convert prediction to class name
    classes = ['Pepper Bell Bacterial Spot', 'Potato Late Blight', 'Tomato Late Blight']
    
    return classes[prediction[0]]

if __name__ == "__main__":
    # Example usage
    image_path = input("Enter the path to your plant image: ")
    if os.path.exists(image_path):
        result = predict_disease(image_path)
        print(f"Predicted Disease: {result}")
    else:
        print("Error: Image file not found")

import os
from PIL import Image
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import joblib
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

# Set the dataset path to the current directory's PlantVillage folder
dataset_path = 'PlantVillage'
selected_classes = ['Pepper__bell___Bacterial_spot', 'Potato___Late_blight', 'Tomato_Late_blight']

data = []
labels = []

# Iterate through the dataset directory
for class_name in os.listdir(dataset_path):
    if class_name in selected_classes:
        class_dir = os.path.join(dataset_path, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            data.append(img_path)
            labels.append(class_name)

df = pd.DataFrame({'data': data, 'label': labels})

def extract_hog_features(image):
    # Convert the image to grayscale using cv2
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor()
    # Compute HOG features
    hog_features = hog.compute(gray_image)
    return hog_features.flatten()

def resize_image(image, new_size=(128, 128)):
    return cv2.resize(image, new_size)

# Process images in batches
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
batch_size = 32
features_list = []
labels_list = []

print("Processing images...")
for start in range(0, len(df_shuffled), batch_size):
    end = min(start + batch_size, len(df_shuffled))
    batch = df_shuffled[start:end]
    
    batch_features = []
    batch_labels = []
    
    for index, row in batch.iterrows():
        image = cv2.imread(row['data'])
        resized_image = resize_image(image)
        hog_features = extract_hog_features(resized_image)
        batch_features.append(hog_features)
        batch_labels.append(row['label'])
    
    features_list.extend(batch_features)
    labels_list.extend(batch_labels)
    print(f"Processed {end}/{len(df_shuffled)} images")

# Convert lists to NumPy arrays
features_array = np.array(features_list)
labels_array = np.array(labels_list)
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels_array)

print("Training model...")
X_train, X_test, y_train, y_test = train_test_split(features_array, labels_encoded, test_size=0.25, random_state=42, stratify=labels_encoded)

lr_pipeline = Pipeline([
    ('pca', PCA(n_components=2100, random_state=42)),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

lr_pipeline.fit(X_train, y_train)

predictions = lr_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model accuracy: {accuracy:.4f}")

# Save the trained model
print("Saving model...")
joblib.dump(lr_pipeline, 'plant_disease_model.joblib')
print("Model saved as 'plant_disease_model.joblib'")

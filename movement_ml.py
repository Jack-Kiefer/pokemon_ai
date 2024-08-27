import json
import numpy as np
import os
from PIL import Image
from tensorflow.keras import models, layers, optimizers
from imblearn.over_sampling import SMOTE  # Import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_data(json_file, folder_path):
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    images = []
    labels = []
    for item in data:
        img1 = Image.open(os.path.join(folder_path, item['image1'])).convert('RGB')
        img2 = Image.open(os.path.join(folder_path, item['image2'])).convert('RGB')
        img1 = np.array(img1.resize((128, 128))) / 255.0
        img2 = np.array(img2.resize((128, 128))) / 255.0
        # Stack images along the channel axis
        stacked_images = np.concatenate((img1, img2), axis=-1)
        images.append(stacked_images)
        labels.append(item['label'])
    
    images = np.array(images)
    labels = np.array(labels)

    # Encode labels to handle multiple classes
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    images_reshaped = images.reshape(images.shape[0], -1)  # Reshape for SMOTE
    images_resampled, labels_resampled = smote.fit_resample(images_reshaped, labels_encoded)
    images_resampled = images_resampled.reshape(-1, 128, 128, 6)  # Reshape back to original

    return images_resampled, labels_resampled, label_encoder

def create_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')  # Use softmax for multi-class
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])  # Use sparse_categorical_crossentropy
    return model

def train_model(images, labels, num_classes):
    model = create_model(images[0].shape, num_classes)
    model.fit(images, labels, epochs=20, validation_split=0.6)
    model.save('movement_detection_model.h5')

def main():
    folder_path = 'screenshots'
    json_file = 'label_data.json'
    images, labels, label_encoder = load_data(json_file, folder_path)

    # Determine the number of classes from the encoder
    num_classes = len(label_encoder.classes_)

    train_model(images, labels, num_classes)

if __name__ == "__main__":
    main()

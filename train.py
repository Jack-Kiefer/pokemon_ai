import json
import numpy as np
import os
from PIL import Image
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

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
        layers.Conv2D(256, (3, 3), activation='relu'),  # Additional convolutional layer
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.005)),  # Increased regularization
        layers.Dropout(0.5),  # Dropout to prevent overfitting
        layers.Dense(num_classes, activation='softmax')  # Output layer for multi-class classification
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(images, labels, num_classes):
    # Split data into training and validation sets using stratified split
    train_images, val_images, train_labels, val_labels = train_test_split(
        images, labels, test_size=0.15, stratify=labels, random_state=42
    )

    # Define data augmentation for training data
    train_datagen = ImageDataGenerator(
        rotation_range=10,  # Reduced from 20
        width_shift_range=0.1,  # Reduced from 0.2
        height_shift_range=0.1,  # Reduced from 0.2
        shear_range=0.1,  # Reduced from 0.2
        zoom_range=0.1,  # Reduced from 0.2
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Define data generator for validation data (without augmentation)
    val_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow(train_images, train_labels, batch_size=32)
    val_generator = val_datagen.flow(val_images, val_labels, batch_size=32)

    model = create_model(train_images[0].shape, num_classes)

    # Define learning rate scheduler
    def scheduler(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * 0.5  # Reduce learning rate less aggressively

    lr_scheduler = LearningRateScheduler(scheduler)

    # Define early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)  # Increased patience

    # Fit the model using generators
    model.fit(
        train_generator,
        epochs=20,
        validation_data=val_generator,
        callbacks=[lr_scheduler, early_stopping]
    )

    model.save('movement_detection_model.h5')
    return model

def evaluate_model(model, images, labels, label_encoder):
    predictions = model.predict(images)
    predicted_classes = np.argmax(predictions, axis=1)
    print(classification_report(labels, predicted_classes, target_names=label_encoder.classes_))
    print(confusion_matrix(labels, predicted_classes))

def main():
    folder_path = 'screenshots'
    json_file = 'label_data.json'
    images, labels, label_encoder = load_data(json_file, folder_path)

    # Determine the number of classes from the encoder
    num_classes = len(label_encoder.classes_)

    model = train_model(images, labels, num_classes)
    evaluate_model(model, images, labels, label_encoder)

if __name__ == "__main__":
    main()

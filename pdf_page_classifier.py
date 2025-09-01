
# PDF Page Classification CNN Model
# Complete implementation for classifying PDF pages as good or bad

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import numpy as np
import os
from PIL import Image

class PDFPageClassifier:
    """
    Complete CNN-based PDF page classifier
    """

    def __init__(self, input_shape=(224, 224, 3), model_type="full"):
        self.input_shape = input_shape
        self.model_type = model_type
        self.model = None
        self.history = None

    def create_model(self):
        """Create CNN model based on specified type"""

        if self.model_type == "full":
            # Full CNN model with more parameters
            self.model = models.Sequential([
                layers.Input(shape=self.input_shape),

                # First convolutional block
                layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),

                # Second convolutional block
                layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),

                # Third convolutional block
                layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),

                # Fourth convolutional block
                layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),

                # Global average pooling
                layers.GlobalAveragePooling2D(),

                # Dense layers
                layers.Dense(512, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(256, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.5),

                # Output layer
                layers.Dense(1, activation='sigmoid')
            ])

        else:  # lightweight model
            self.model = models.Sequential([
                layers.Input(shape=self.input_shape),

                # First block
                layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),

                # Second block
                layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),

                # Third block
                layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),

                # Global pooling and output
                layers.GlobalAveragePooling2D(),
                layers.Dense(128, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(1, activation='sigmoid')
            ])

        # Compile model
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )

        return self.model

    def create_data_generators(self, train_dir, validation_dir, batch_size=32):
        """Create data generators for training and validation"""

        # Training data generator with augmentation
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )

        # Validation data generator
        validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

        # Create generators
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='binary',
            classes=['bad', 'good'],
            shuffle=True
        )

        validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='binary',
            classes=['bad', 'good'],
            shuffle=False
        )

        return train_generator, validation_generator

    def train(self, train_generator, validation_generator, epochs=50, save_path='best_pdf_classifier.h5'):
        """Train the model"""

        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")

        # Define callbacks
        callbacks_list = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                save_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
        ]

        # Train the model
        self.history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=callbacks_list,
            verbose=1
        )

        return self.history

    def preprocess_image(self, image_path):
        """Preprocess single image for prediction"""
        try:
            # Load and preprocess image
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')

            image = image.resize(self.input_shape[:2], Image.Resampling.LANCZOS)
            image_array = np.array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0)

            return image_array
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {e}")
            return None

    def predict(self, image_path, threshold=0.5):
        """Predict single image"""
        if self.model is None:
            raise ValueError("Model not loaded. Create or load a model first.")

        processed_image = self.preprocess_image(image_path)
        if processed_image is None:
            return None, None, None

        prediction_prob = self.model.predict(processed_image, verbose=0)[0][0]
        is_good = prediction_prob >= threshold
        classification = "good" if is_good else "bad"
        confidence = prediction_prob if is_good else 1 - prediction_prob

        return classification, confidence, prediction_prob

    def predict_batch(self, image_paths, threshold=0.5):
        """Predict multiple images"""
        results = []
        for image_path in image_paths:
            result = self.predict(image_path, threshold)
            results.append({
                'image_path': image_path,
                'classification': result[0],
                'confidence': result[1],
                'probability': result[2]
            })
        return results

    def load_model(self, model_path):
        """Load trained model"""
        self.model = tf.keras.models.load_model(model_path)
        return self.model

    def save_model(self, model_path):
        """Save trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        self.model.save(model_path)

# Usage example and setup instructions
def create_directory_structure():
    """Create recommended directory structure"""
    directories = [
        'data/train/good',
        'data/train/bad',
        'data/validation/good',
        'data/validation/bad',
        'data/test/good',
        'data/test/bad',
        'models',
        'predictions'
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

# Alternative approaches for PDF classification
def create_transfer_learning_model(base_model_name='ResNet50'):
    """Create transfer learning model using pre-trained networks"""

    if base_model_name == 'ResNet50':
        base_model = tf.keras.applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
    elif base_model_name == 'MobileNetV2':
        base_model = tf.keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )

    # Freeze base model layers
    base_model.trainable = False

    # Add custom classifier
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )

    return model

# Main execution example
if __name__ == "__main__":
    # Create classifier
    classifier = PDFPageClassifier(model_type="lightweight")  # or "full"

    # Create model
    # model = classifier.create_model()
    # print("Model created successfully!")
    
    model_path = "best_pdf_classifier.h5"
    model = classifier.load_model(model_path)
    print("Model loaded successfully!")

    # Example training (uncomment when you have data)
    # train_gen, val_gen = classifier.create_data_generators('data/train', 'data/validation')
    # history = classifier.train(train_gen, val_gen, epochs=50)

    # Example prediction (uncomment when you have a trained model)
    classification, confidence, probability = classifier.predict('data/train/good/117.png')
    print(f"Page classified as: {classification} (confidence: {confidence:.2f})")

print("PDF Page Classifier implementation complete!")

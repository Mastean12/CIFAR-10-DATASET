# Import necessary libraries
from tensorflow.keras.datasets import cifar10  # CIFAR-10 dataset
from tensorflow import keras  # Core Keras library for neural networks
from tensorflow.keras import layers  # Layers module for creating the model

import numpy as np  # NumPy for numerical operations
import os  # OS module for environment variables

# Disable oneDNN custom operations to avoid numerical differences warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load the CIFAR-10 dataset from Keras
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Print shapes of training and testing data to understand the dimensions
print("Training data shape:", train_images.shape)  # (50000, 32, 32, 3)
print("Training labels shape:", train_labels.shape)  # (50000, 1)
print("Test data shape:", test_images.shape)  # (10000, 32, 32, 3)
print("Test labels shape:", test_labels.shape)  # (10000, 1)

# Normalize the image data to be in the range of 0 to 1 by dividing by 255.0
train_images = train_images.astype("float32") / 255
test_images = test_images.astype("float32") / 255

# Flatten the images to a 1D vector (3072 = 32*32*3) for input into the Dense layer
train_images = train_images.reshape((50000, 32 * 32 * 3))
test_images = test_images.reshape((10000, 32 * 32 * 3))

# Print the new shape of training and testing images to verify flattening
print("Flattened training data shape:", train_images.shape)  # (50000, 3072)
print("Flattened test data shape:", test_images.shape)  # (10000, 3072)

# Define the model architecture
model = keras.Sequential([
    layers.Input(shape=(32 * 32 * 3,)),  # Specify input shape as a tuple to avoid warning
    layers.Dense(512, activation="relu"),  # Hidden layer with 512 neurons and ReLU activation
    layers.Dense(10, activation="softmax")  # Output layer with 10 neurons (one for each class) and softmax activation
])

# Compile the model with optimizer, loss function, and metrics
model.compile(optimizer="rmsprop",  # RMSprop optimizer
              loss="sparse_categorical_crossentropy",  # Loss function for integer labels
              metrics=["accuracy"])  # Metric to evaluate during training and testing

# Train the model on the training data
model.fit(train_images, train_labels, epochs=5, batch_size=128)  # 5 epochs and batch size of 128

# Evaluate the model on the test data to compute the loss and accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc:.4f}")  # Print test accuracy

# Make predictions on the first 10 test images
test_digits = test_images[0:10]  # Select the first 10 images from the test set
predictions = model.predict(test_digits)  # Get the model's predictions for these images

# Print the predicted class for the first image in the test set
print("Predicted class for the first test image:", predictions[0].argmax())
print("Actual class for the first test image:", test_labels[0][0])  # Print the actual class for comparison

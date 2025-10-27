import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Set image path and class labels
img_path = '' # Path the Test image is Saved in
classes = ['cardboard', 'plastic', 'glass', 'metal', 'paper', 'trash']

# Check if image exists
print(f"Checking if image exists at path: {img_path}")
if not os.path.exists(img_path):
    print("Image file not found.")
    exit()

# Load the trained model
model_path = '' Path the Model is Saved in
print("Loading model...")
model = load_model(model_path)
print("Model loaded successfully.")

# Load and preprocess the image
img = image.load_img(img_path, target_size=(128, 128))  # Size should match training input
img_array = image.img_to_array(img) / 255.0  # Normalize
img_array_expanded = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Make prediction
predictions = model.predict(img_array_expanded)
print(f"\nPredictions (raw probabilities):\n{predictions}\n")

# Output class probabilities
print("Class Probabilities:")
for cls, prob in zip(classes, predictions[0]):
    print(f"- {cls}: {int(prob * 100)}%")

# Identify predicted class
predicted_class = classes[np.argmax(predictions[0])]
print(f"\nPredicted Class: {predicted_class}")

# Display image with predicted label
img_display = image.load_img(img_path, target_size=(224, 224))
img_display_array = image.img_to_array(img_display).astype("uint8") / 255.0  # Normalize for display

plt.imshow(img_display_array)
plt.title("Test Image")
plt.axis("off")
plt.text(10, 210, f"Predicted Class: {predicted_class}", color="white",
         fontsize=12, bbox=dict(facecolor='black', alpha=0.7))
plt.show()

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Debug: List folder contents
print("Checking contents of 'Path the trashnet_dataset Saved in':")
for filename in os.listdir('Path the trashnet_dataset Saved in'):
    print(" -", filename)

# Parameters
img_size = 128
batch_size = 32

# Data generator with validation split
datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)

val_data = datagen.flow_from_directory(
    'model/trashnet_dataset',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Load trained model
model = load_model('Path the Model is Saved in')

# Predict
predictions = model.predict(val_data, batch_size=batch_size)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = val_data.classes
class_labels = list(val_data.class_indices.keys())

# Classification report
print("\nClassification Report:")
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

# Confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels, cmap="Blues")
plt.title("Confusion Matrix")
plt.ylabel('Actual')
plt.xlabel('Predicted') 
plt.tight_layout()
plt.show()

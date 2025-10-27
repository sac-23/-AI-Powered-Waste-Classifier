
#  AI-Powered Waste Classifier

**Waste Classifier AI** is a Convolutional Neural Network (CNN)-based image classification system built using **TensorFlow/Keras**.  
It automatically classifies waste materials such as **cardboard, plastic, glass, metal, paper, and trash**, promoting efficient recycling and sustainable waste management.


## Overview

This AI model learns to distinguish different waste types from image data.  
It uses **supervised learning** on a dataset of categorized waste images and predicts the correct category for new images.  
The model can be integrated into **smart waste bins** or **recycling systems** for automated sorting.


## Features

-  Multi-class waste classification using CNN  
-  Data augmentation for improved generalization  
-  Model evaluation using confusion matrix and classification report  
-  Real-time image prediction via Python script  
-  Easy-to-train and deploy deep learning architecture  


## Model Architecture

The CNN is designed with the following layers:

1. **Conv2D + ReLU** – Feature extraction from input images  
2. **MaxPooling2D** – Downsampling to reduce spatial dimensions  
3. **Conv2D (64 filters)** – Deeper feature extraction  
4. **MaxPooling2D**  
5. **Conv2D (128 filters)** – High-level feature learning  
6. **Flatten + Dropout (0.5)** – Flattening and regularization  
7. **Dense (128 neurons)** – Fully connected layer  
8. **Output Layer (Softmax)** – Predicts one of six classes  

**Loss Function:** Categorical Crossentropy  
**Optimizer:** Adam  
**Metrics:** Accuracy  


## How It Works

1. Images are preprocessed and normalized to a **128×128×3** format.  
2. The CNN learns visual patterns specific to each waste type.  
3. During prediction, the trained model classifies a new image into one of six waste categories.  
4. Model performance is evaluated using **accuracy metrics** and a **confusion matrix**.


## Technologies Used

- Python 3.10+  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  
- Seaborn  
- scikit-learn  

## Dataset

The dataset contains **six waste categories** used for training and validation:

- Cardboard  
- Glass  
- Metal  
- Paper  
- Plastic  
- Trash

Each folder inside the dataset directory contains labeled images for that specific waste type.


# AI Skin Disease Detection & Medical Assistant

## Overview

AI Skin Disease Detection & Medical Assistant is a multimodal deep learning project that combines:

- Transfer Learning
- Computer Vision
- Generative AI
- Medical Image Classification
- Gemini Vision
- Interactive Streamlit UI

The system predicts skin diseases from uploaded images using a fine-tuned MobileNetV2 model and provides contextual AI-based medical explanations using Gemini AI.

---

# Features

## Skin Disease Detection
- CNN-based skin disease classification
- Transfer Learning using MobileNetV2
- Fine-tuned deep learning model
- Confidence score prediction
- Multi-class disease detection

---

## AI Medical Assistant
- Context-aware AI chatbot
- Understands uploaded image context
- Responds according to prediction and image analysis
- Explains symptoms and precautions
- Provides medical guidance

---

## Gemini Vision Integration
The system can analyze:
- lesion patterns
- pigmentation
- texture
- asymmetry
- visual abnormalities

from uploaded skin images.

---

## Streamlit Interface
- Drag-and-drop image upload
- Instant prediction
- Real-time AI chatbot
- Interactive medical dashboard

---

# Supported Classes

| Class | Disease |
|------|------|
| akiec | Actinic Keratoses |
| bcc | Basal Cell Carcinoma |
| bkl | Benign Keratosis |
| df | Dermatofibroma |
| mel | Melanoma |
| nv | Melanocytic Nevi |
| vasc | Vascular Lesions |

---

# Dataset

Dataset Used:
## HAM10000 Skin Lesion Dataset

Download:
https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

---

# Tech Stack

## Frontend
- Streamlit

## Deep Learning
- TensorFlow
- Keras
- MobileNetV2

## Generative AI
- Google Gemini API
- Gemini Vision

## Data Processing
- NumPy
- Pandas
- PIL

---

# Transfer Learning

This project uses MobileNetV2 pretrained on ImageNet.

Benefits:
- Faster training
- Better feature extraction
- Reduced overfitting
- Improved accuracy

Fine tuning is applied to improve medical image classification performance.

---

# Model Architecture

- MobileNetV2 Base Model
- GlobalAveragePooling2D
- BatchNormalization
- Dense Layers
- Dropout Regularization
- Softmax Output Layer

---

# Data Augmentation

The following augmentation techniques are used:
- Rotation
- Zoom
- Horizontal Flip
- Brightness Adjustment
- Width & Height Shift

---

# AI Workflow

## Step 1
User uploads skin image.

## Step 2
CNN predicts disease class.

## Step 3
Gemini Vision analyzes image characteristics.

## Step 4
Prediction + image interpretation are passed to AI assistant.

## Step 5
User interacts with contextual chatbot.

---

# Project Structure

```text
Skin_Disease_Detection/
│
├── all_images/
│
├── app/
│   ├── .env
│   └── Main.py
│
├── dataset/
│   └── HAM10000_metadata.csv
│
├── datasett/
│   ├── train/
│   └── validation/
│
├── model/
│   ├── skin_disease_model.h5
│   └── train.py
│
├── EDA.ipynb
│
├── organize_dataset.py
│
├── requirements.txt
│
└── README.md
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/DeepanshuKumar09/AI-Powered-Skin-Diagnosis-Tool
cd Skin_Disease_Detection
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# AI Chatbot Capabilities

Users can ask:
- Is this dangerous?
- What are the symptoms?
- Should I consult a doctor?
- Can this spread?
- What precautions should I take?
- Why did the model predict this?

The chatbot responds according to:
- uploaded image
- CNN prediction
- confidence score
- Gemini visual analysis

---

# Future Features

- Real-time camera-based skin detection
- Grad-CAM visualization for Explainable AI
- Voice-enabled AI assistant
- Multi-language chatbot support
- PDF medical report generation
- Mobile app deployment
- EfficientNet and Vision Transformer integration
- User authentication and patient history tracking

---

# Disclaimer

This project is developed for educational and research purposes only.

---

# Author
Deepanshu Kumar

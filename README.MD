<<<<<<< HEAD
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
=======
# DermAI: AI-Powered Skin Diagnostics & Medical Assistant Dashboard

DermAI is a clinical-grade multimodal intelligence dashboard that combines **Computer Vision (CNNs)**, **Explainable AI (XAI)**, and **Generative AI (LLMs)** to aid in skin disease classification, explanation, and patient education.

The application leverages a fine-tuned **MobileNetV2** convolutional neural network to predict skin anomalies from images, maps decision-making paths using **Grad-CAM class activation maps**, and delivers context-aware, multilingual explanations using the **Google Gemini API**.

---

## 🌟 Key Features

### 1. 🏠 Diagnostic Center
*   **Dual-Source Input:** Upload high-resolution images (`.jpg`, `.jpeg`, `.png`) or capture a skin lesion in real-time using your live webcam.
*   **CNN Predictions:** Automated multi-class predictions mapping to HAM10000 diagnostic criteria, accompanied by confidence percentages.
*   **AI Interpretation:** Multimodal visual analysis of texture, borders, and margins provided in your preferred language (English, Spanish, Hindi, French, Arabic, Portuguese).
*   **Stateful Chatbot:** A fully conversational, context-aware AI assistant to answer follow-up questions regarding symptoms, spread, and care.
*   **Clinical Summaries:** Instantly compile and download a formatted PDF medical report of your diagnostic session.

### 2. 🔍 Explainable AI (Grad-CAM)
*   **Gradient Heatmaps:** Computes gradients flowing into the final convolutional layer (`out_relu`) of the CNN model.
*   **Visual Explanations:** Overlays a class-activation heatmap on top of the original skin image, letting clinicians see the precise pixels the model focused on.
*   **Interactive Opacity:** Adjust the transparency of the overlay dynamically using a slider.

### 3. 📊 Patient Analytics Dashboard
*   **Clinical Insights:** Parses and visualizes demographic data from the HAM10000 dataset (10,000+ patient records).
*   **Interactive Visualizations:** Powered by Plotly Express:
    *   **Prevalence:** Overall diagnosis counts.
    *   **Anatomy:** Distribution of lesions by body site.
    *   **Demographics:** Age group distributions parsed by gender (box plots and pie charts).

### 4. 📚 Dermatology Encyclopedia
*   **Searchable Knowledge Base:** Comprehensive references for all 7 supported disease categories.
*   **Detailed Information:** Clear explanations of pathological descriptions, common symptoms, safety precautions, and standardized recommended clinical actions.

### 5. ⚕️ FAQ & ABCDE self-screening
*   **Dermatologist Guidelines:** Guides users through the ABCDE screening method for Melanoma (Asymmetry, Border, Color, Diameter, Evolving).
*   **Skin Health FAQ:** Explanations covering UV exposure, skin checking frequencies, SPF recommendations, and AI error rates.

---

## 🧬 Supported Classifications

The underlying MobileNetV2 CNN categorizes skin lesions into seven distinct classes:

| Code | Pathology | Classification Type | Severity Level |
|---|---|---|---|
| **akiec** | Actinic Keratosis / Intraepithelial Carcinoma | Pre-cancerous | Moderate |
| **bcc** | Basal Cell Carcinoma | Malignant (Cancerous) | High |
| **bkl** | Benign Keratosis (Seborrheic / Lichen Planus) | Benign (Non-cancerous) | Low |
| **df** | Dermatofibroma | Benign (Non-cancerous) | Low |
| **mel** | Melanoma | Malignant (Aggressive) | Critical |
| **nv** | Melanocytic Nevi (Common Mole) | Benign (Non-cancerous) | Low |
| **vasc** | Vascular Lesion (Cherry Angioma, etc.) | Benign (Non-cancerous) | Low |

---

## 🛠️ Tech Stack

*   **Frontend UI:** [Streamlit](https://streamlit.io/) (including chat elements, camera capture, and sidebar navigation).
*   **Deep Learning:** [TensorFlow 2.15](https://www.tensorflow.org/) & [Keras](https://keras.io/) (MobileNetV2 CNN classifier).
*   **Generative AI:** [Google Gemini API](https://ai.google.dev/) (`gemini-2.5-flash` model).
*   **Explainable AI:** Custom Grad-CAM activation mapping with [OpenCV](https://opencv.org/) and NumPy.
*   **Data Analytics:** [Pandas](https://pandas.pydata.org/) and [Plotly Express](https://plotly.com/python/plotly-express/).

---

## 🚀 Installation & Local Run

### 1. Clone & Set Up Directory
```bash
git clone <repository_url>
cd Skin_Disease_Detection
```

### 2. Install Dependencies
Make sure you have Python 3.10+ installed. Install the requirements:
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the root of the project (or inside `app/`) and append your Gemini API key:
```env
GEMINI_API_KEY="your_api_key_here"
```
*(If no environment key is found, the system will fall back to a default configuration).*

### 4. Run the Dashboard
Start the local development server:
```bash
streamlit run app/Main.py
```
Open your browser and navigate to the address shown in the terminal (usually `http://localhost:8501`).

---

## 📂 Project Structure

```text
Skin_Disease_Detection/
├── app/
│   └── Main.py               # Main multi-page Streamlit application & styling
├── dataset/
│   └── HAM10000_metadata.csv # Clinical dataset metadata
├── model/
│   ├── skin_disease_model.h5 # Fine-tuned MobileNetV2 Keras model
│   └── train.py              # CNN transfer-learning code
├── requirements.txt          # Python dependencies list
└── README.md                 # Project Overview & Quick Start
```

---

## ⚖️ Disclaimer

This project is built for educational and clinical research purposes only. The predictions, heatmaps, and AI-generated text do not constitute professional medical advice, diagnosis, or treatment. Always consult a board-certified dermatologist for any skin concerns.
>>>>>>> 46bee89 (Updated feature)

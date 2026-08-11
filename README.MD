AI-Powered Skin Diagnostics & Medical Assistant Dashboard

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

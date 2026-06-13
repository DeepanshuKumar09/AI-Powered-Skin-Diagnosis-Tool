import streamlit as st
import tensorflow as tf
import numpy as np
import google.generativeai as genai
#from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from dotenv import load_dotenv
load_dotenv()

vision_model = genai.GenerativeModel("gemini-2.5-flash")

model = tf.keras.models.load_model(
    "C:/Projects/Langchain_Model/Skin_Disease_Detection/model/skin_disease_model.h5"
)

classes = [
    'akiec',
    'bcc',
    'bkl',
    'df',
    'mel',
    'nv',
    'vasc'
]

st.set_page_config(
    page_title="AI Skin Disease Assistant",
    layout="wide"
)

st.title("AI Skin Disease Detection + Medical Assistant")

if "analysis" not in st.session_state:
    st.session_state.analysis = ""

if "prediction" not in st.session_state:
    st.session_state.prediction = ""

if "confidence" not in st.session_state:
    st.session_state.confidence = 0


uploaded_file = st.file_uploader(
    "Upload Skin Image",
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Skin Image", width=350)

    cnn_image = image.resize((224,224))
    cnn_image = np.array(cnn_image)
    cnn_image = preprocess_input(cnn_image)
    cnn_image = np.expand_dims(cnn_image, axis=0)
    prediction = model.predict(cnn_image)
    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.session_state.prediction = predicted_class
    st.session_state.confidence = confidence
    
    st.subheader("CNN Prediction")
    st.write(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")

    with st.spinner("Analyzing image with Gemini Vision..."):

        vision_prompt = """
        Analyze this skin disease image.
        Explain:
        - possible visual symptoms
        - color patterns
        - lesion characteristics
        - texture
        - abnormal findings
        Keep explanation medical but simple.
        """
        response = vision_model.generate_content([
            vision_prompt,
            image
        ])

        st.session_state.analysis = response.text

    st.subheader("AI Image Interpretation")

    st.write(st.session_state.analysis)

st.header("Chat with AI Medical Assistant")
user_question = st.text_area(
    "Ask questions about uploaded image"
)

if st.button("Ask AI"):
    if uploaded_file is not None:
        full_prompt = f"""
        You are an AI medical assistant.
        Uploaded image CNN prediction:
        {st.session_state.prediction}
        Prediction confidence:
        {st.session_state.confidence:.2f}%
        Gemini visual analysis:
        {st.session_state.analysis}

        User Question:
        {user_question}
        Give:
        - medically relevant explanation
        - beginner friendly answer
        - precautions if needed
        - possible symptoms
        - recommendation to consult doctor if necessary
        Do not give dangerous medical claims.
        """

        final_response = vision_model.generate_content(full_prompt)
        st.subheader("AI Response")
        st.write(final_response.text)

    else:
        st.warning("Please upload an image first.")

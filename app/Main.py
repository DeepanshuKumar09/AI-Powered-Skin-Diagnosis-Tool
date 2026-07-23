import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import google.generativeai as genai
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from dotenv import load_dotenv
import cv2
import os
from datetime import datetime
import plotly.express as px

# 1. LOAD CONFIGURATION & ENVIRONMENT
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY", "")
if api_key:
    genai.configure(api_key=api_key)
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

from dotenv import load_dotenv
import cv2
import os
from datetime import datetime
import plotly.express as px

# 1. LOAD CONFIGURATION & ENVIRONMENT
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    # Use fallback hardcoded key for continuity
    api_key = "<YOUR_GEMINI_API_KEY>"
genai.configure(api_key=api_key)
vision_model = genai.GenerativeModel("gemini-2.5-flash")

# 2. PAGE CONFIGURATION
st.set_page_config(
    page_title="DermAI - Advanced Skin Diagnostics Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 3. CUSTOM STYLING (CSS)
st.markdown("""
<style>
    /* Styling headers */
    .main-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        color: #3B82F6;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }
    .section-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        color: #60A5FA;
        font-weight: 600;
        margin-top: 15px;
        margin-bottom: 10px;
    }
    /* Card style for stats and info */
    .metric-card {
        background-color: #1E293B;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        border-left: 5px solid #3B82F6;
        border-top: 1px solid #334155;
        border-right: 1px solid #334155;
        border-bottom: 1px solid #334155;
        margin-bottom: 15px;
    }
    .metric-title {
        font-size: 14px;
        color: #94A3B8;
        font-weight: 500;
        margin-bottom: 5px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 26px;
        color: #F8FAFC;
        font-weight: 700;
    }
    .encyclopedia-card {
        background-color: #1E293B;
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 20px;
        border: 1px solid #334155;
        color: #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .encyclopedia-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
    }
    /* Severity badges */
    .badge-benign {
        background-color: #14532D;
        color: #86EFAC;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
        border: 1px solid #166534;
    }
    .badge-precancerous {
        background-color: #713F12;
        color: #FDE047;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
        border: 1px solid #854D0E;
    }
    .badge-malignant {
        background-color: #7F1D1D;
        color: #FCA5A5;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
        border: 1px solid #991B1B;
    }
    .badge-critical {
        background-color: #450A0A;
        color: #FECACA;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
        border: 1px solid #7F1D1D;
    }
    /* Sidebar styling refinement */
    .sidebar-header {
        font-size: 20px;
        font-weight: 700;
        color: #60A5FA;
        text-align: center;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# 4. DATASET & MODELS CACHING
@st.cache_resource
def load_cnn_model():
    model_path = "C:/Projects/Langchain_Model/Skin_Disease_Detection/model/skin_disease_model.h5"
    return tf.keras.models.load_model(model_path)

@st.cache_data
def load_metadata():
    csv_path = "C:/Projects/Langchain_Model/Skin_Disease_Detection/dataset/HAM10000_metadata.csv"
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
        dx_map = {
            'akiec': 'Actinic Keratosis',
            'bcc': 'Basal Cell Carcinoma',
            'bkl': 'Benign Keratosis',
            'df': 'Dermatofibroma',
            'mel': 'Melanoma',
            'nv': 'Melanocytic Nevi',
            'vasc': 'Vascular Lesion'
        }
        df['disease_name'] = df['dx'].map(dx_map)
        df['age'] = df['age'].fillna(df['age'].median())
        return df
    except Exception as e:
        print(f"Error loading CSV metadata: {e}")
        return None

# Load resources
try:
    model = load_cnn_model()
except Exception as e:
    st.error(f"Error loading AI model: {e}. Please check model path.")

# 5. REFERENCE DICTIONARY
classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

dx_info = {
    'akiec': {
        'full_name': 'Actinic Keratosis / Intraepithelial Carcinoma',
        'type': 'Pre-cancerous',
        'severity': 'Moderate',
        'badge': 'precancerous',
        'description': 'Rough, scaly patches on skin areas exposed to the sun (like face, lips, ears, back of hands). It is a precursor to squamous cell carcinoma.',
        'symptoms': 'Flat to slightly raised scaly or crusty surface, dry or rough texture, itching or burning sensation, pink, red, or brown coloration.',
        'precautions': 'Strict UV protection: minimize sun exposure between 10 AM and 4 PM, apply broad-spectrum SPF 30+ sunscreen daily, wear protective clothing.',
        'action': 'Consult a dermatologist for screening. Options like cryotherapy, laser therapy, or topical prescription creams are standard.'
    },
    'bcc': {
        'full_name': 'Basal Cell Carcinoma',
        'type': 'Malignant (Skin Cancer)',
        'severity': 'High',
        'badge': 'malignant',
        'description': 'The most common form of skin cancer. Usually occurs on sun-exposed areas. It grows slowly and rarely spreads to other parts of the body, but can cause local damage if left untreated.',
        'symptoms': 'A pearly or waxy bump, a flat flesh-colored or brown scar-like lesion, or a bleeding/scabbing sore that heals and returns.',
        'precautions': 'Protect skin from UV radiation, avoid tanning beds, perform monthly skin checks.',
        'action': 'Requires formal medical evaluation and biopsy. Treatment options include minor surgical excision, Mohs surgery, or freezing (cryosurgery).'
    },
    'bkl': {
        'full_name': 'Benign Keratosis (Seborrheic / Lichen Planus-like)',
        'type': 'Benign (Non-cancerous)',
        'severity': 'Low',
        'badge': 'benign',
        'description': 'Very common non-cancerous skin growths, often appearing in older adults. They are benign and not contagious.',
        'symptoms': 'Waxy, scaly, or crusty appearance, round or oval shape, characteristic "stuck-on" look, colors range from light tan to black.',
        'precautions': 'Generally harmless. Avoid scratching, picking, or rubbing the growth to prevent irritation and secondary infection.',
        'action': 'No medical treatment is required. Removal is optional if the lesion becomes irritated by clothing, itchy, or for cosmetic preference.'
    },
    'df': {
        'full_name': 'Dermatofibroma',
        'type': 'Benign (Non-cancerous)',
        'severity': 'Low',
        'badge': 'benign',
        'description': 'Common, firm, harmless bumps under the skin, most frequently found on the lower legs of adults.',
        'symptoms': 'Small, firm red-to-brown nodule, often slightly tender or itchy. It displays the "dimple sign" (dimples inward when pinched).',
        'precautions': 'No special precautions required. Monitor for rapid size increases or ulceration.',
        'action': 'Usually left alone. If painful, cosmetically undesirable, or if diagnosis is uncertain, it can be surgically removed.'
    },
    'mel': {
        'full_name': 'Melanoma',
        'type': 'Malignant (Aggressive Skin Cancer)',
        'severity': 'Critical',
        'badge': 'critical',
        'description': 'The most serious type of skin cancer, originating in melanocytes. It can spread quickly (metastasize) to other organs if not detected and treated early.',
        'symptoms': 'Asymmetrical border, irregular edges, color variations within the same lesion, diameter greater than 6mm, or evolving over time (ABCDE rule).',
        'precautions': 'Avoid direct sunlight, wear SPF 50+ sunscreen, wear protective clothing, and inspect skin regularly.',
        'action': 'Urgent dermatological consultation and biopsy. Early-stage melanoma is highly curable via surgical excision.'
    },
    'nv': {
        'full_name': 'Melanocytic Nevi (Common Moles)',
        'type': 'Benign (Non-cancerous)',
        'severity': 'Low',
        'badge': 'benign',
        'description': 'Common benign skin growths caused by clusters of melanocytes. Most moles are completely harmless.',
        'symptoms': 'Uniform tan, brown, or black color, round or oval shape, flat or dome-shaped, distinct and regular borders.',
        'precautions': 'Monitor regular moles monthly using the ABCDE guidelines. Note any changes in size, shape, color, or texture.',
        'action': 'No action is needed. Consult a dermatologist if a mole develops irregular features, grows rapidly, bleeds, or itches.'
    },
    'vasc': {
        'full_name': 'Vascular Lesions (Cherry Angiomas / Pyogenic Granulomas)',
        'type': 'Benign (Non-cancerous)',
        'severity': 'Low',
        'badge': 'benign',
        'description': 'Benign abnormalities of the skin blood vessels, including cherry angiomas, angiokeratomas, and pyogenic granulomas.',
        'symptoms': 'Bright red, purple, or blue spots or bumps, smooth or rough texture, cherry angiomas are small and painless but pyogenic granulomas can bleed easily.',
        'precautions': 'Avoid picking or scratching to prevent bleeding. Protect from friction.',
        'action': 'Usually harmless. Pyogenic granulomas are often removed due to frequent bleeding. Cherry angiomas can be removed via laser or electrocautery if desired.'
    }
}

# 6. GRAD-CAM (EXPLAINABLE AI) UTILITIES
def generate_gradcam(img_array, model, last_conv_layer_name="out_relu"):
    base_model = model.layers[0]
    try:
        last_conv_layer = base_model.get_layer(last_conv_layer_name)
    except ValueError:
        # Fallback to Conv_1 if out_relu not found
        last_conv_layer = base_model.get_layer("Conv_1")
        
    grad_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=[last_conv_layer.output, base_model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, base_outputs = grad_model(img_array)
        tape.watch(conv_outputs)
        
        # Apply the sequential top layers on base outputs
        x = base_outputs
        for layer in model.layers[1:]:
            x = layer(x, training=False)
            
        pred_idx = tf.argmax(x[0])
        class_channel = x[:, pred_idx]
        
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    return heatmap.numpy()

def overlay_gradcam(original_image, heatmap, alpha=0.45, colormap=cv2.COLORMAP_JET):
    img = np.array(original_image)
    h, w, c = img.shape
    
    # Resize heatmap to match original image dimensions
    heatmap_resized = cv2.resize(heatmap, (w, h))
    
    # Scale heatmap to [0, 255]
    heatmap_scaled = np.uint8(255 * heatmap_resized)
    
    # Apply colormap
    heatmap_color = cv2.applyColorMap(heatmap_scaled, colormap)
    
    # Convert color space for PIL (BGR -> RGB)
    heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # Blend images
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap_color_rgb, alpha, 0)
    return superimposed_img

if "analysis" not in st.session_state:
    st.session_state.analysis = ""


# 6.5 PDF REPORT UTILITY
def generate_pdf_report(case_id, source, language, prediction_name, prediction_code, confidence, type_label, severity, action, analysis):
    from io import BytesIO
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=40, leftMargin=40,
                            topMargin=40, bottomMargin=40)
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor('#1E3A8A'),
        spaceAfter=15
    )
    
    h2_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#2563EB'),
        spaceBefore=12,
        spaceAfter=8
    )
    
    body_style = ParagraphStyle(
        'ReportBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=6
    )
    
    bold_body_style = ParagraphStyle(
        'BoldReportBody',
        parent=body_style,
        fontName='Helvetica-Bold'
    )
    
    disclaimer_style = ParagraphStyle(
        'DisclaimerBody',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#64748B'),
        spaceBefore=15
    )

    story = []
    
    # Title
    story.append(Paragraph("DermAI Clinical Case Report", title_style))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", body_style))
    story.append(Spacer(1, 15))
    
    # Case Information Table
    data = [
        [Paragraph("<b>Parameter</b>", bold_body_style), Paragraph("<b>Value</b>", bold_body_style)],
        [Paragraph("Case ID", body_style), Paragraph(case_id, body_style)],
        [Paragraph("Input Source", body_style), Paragraph(source, body_style)],
        [Paragraph("Preferred Language", body_style), Paragraph(language, body_style)],
    ]
    t = Table(data, colWidths=[150, 350])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (1,0), colors.HexColor('#F1F5F9')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(Paragraph("1. Case Metadata", h2_style))
    story.append(t)
    story.append(Spacer(1, 10))
    
    # CNN Model Predictions
    data_pred = [
        [Paragraph("<b>Diagnostic Variable</b>", bold_body_style), Paragraph("<b>Result</b>", bold_body_style)],
        [Paragraph("Predicted Condition", body_style), Paragraph(f"<b>{prediction_name}</b> ({prediction_code})", body_style)],
        [Paragraph("CNN Confidence", body_style), Paragraph(f"{confidence:.2f}%", body_style)],
        [Paragraph("Pathological Type", body_style), Paragraph(type_label, body_style)],
        [Paragraph("Clinical Severity", body_style), Paragraph(severity, body_style)],
        [Paragraph("Recommended Action", body_style), Paragraph(action, body_style)],
    ]
    t_pred = Table(data_pred, colWidths=[150, 350])
    t_pred.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (1,0), colors.HexColor('#F1F5F9')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(Paragraph("2. CNN Model Predictions", h2_style))
    story.append(t_pred)
    story.append(Spacer(1, 10))
    
    # Gemini Analysis
    story.append(Paragraph("3. Generative AI (Gemini) Visual Interpretation", h2_style))
    
    import re
    paragraphs = analysis.split("\n")
    for p_text in paragraphs:
        p_text = p_text.strip()
        if not p_text:
            continue
            
        is_bullet = False
        is_h3 = False
        is_h2 = False
        
        if p_text.startswith("### "):
            is_h3 = True
            p_text = p_text[4:].strip()
        elif p_text.startswith("## "):
            is_h2 = True
            p_text = p_text[3:].strip()
        elif p_text.startswith("# "):
            is_h2 = True
            p_text = p_text[2:].strip()
        elif p_text.startswith("- ") or p_text.startswith("* "):
            is_bullet = True
            p_text = p_text[2:].strip()

        # HTML escape special XML characters first
        p_text = p_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        
        # Replace markdown formatting with HTML tags supported by ReportLab
        p_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', p_text)
        p_text = re.sub(r'__(.*?)__', r'<b>\1</b>', p_text)
        p_text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', p_text)
        p_text = re.sub(r'_(.*?)_', r'<i>\1</i>', p_text)
        
        if is_bullet:
            p_text = "&bull; " + p_text
            
        curr_style = h2_style if (is_h2 or is_h3) else body_style
        
        try:
            story.append(Paragraph(p_text, curr_style))
        except Exception:
            # Fallback: strip tags if ReportLab encounters any XML parse anomaly
            clean_text = re.sub(r'<[^>]+>', '', p_text)
            if is_bullet:
                clean_text = "&bull; " + clean_text
            story.append(Paragraph(clean_text, curr_style))
            
        story.append(Spacer(1, 4))
            
    story.append(Spacer(1, 10))
    disclaimer_text = ("<b>Disclaimer:</b> This report was compiled by an AI application combining a MobileNetV2 CNN classifier "
                       "and Google Gemini. It is for informational and educational purposes only. It is NOT a professional medical "
                       "diagnosis or consultation. Please consult a board-certified dermatologist for diagnosis and care.")
    story.append(Paragraph(disclaimer_text, disclaimer_style))
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# 7. SESSION STATE INITIALIZATION
if "prediction" not in st.session_state:
    st.session_state.prediction = ""
if "confidence" not in st.session_state:
    st.session_state.confidence = 0.0
if "analysis" not in st.session_state:
    st.session_state.analysis = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_image_hash" not in st.session_state:
    st.session_state.current_image_hash = None


# 8. SIDEBAR NAVIGATION & SETTINGS
st.sidebar.markdown("<div class='sidebar-header'>🩺 DermAI Console</div>", unsafe_allow_html=True)
page = st.sidebar.radio(
    "Navigation Menu", 
    [
        "🏠 Diagnostic Center", 
        "📊 Patient Analytics Dashboard", 
        "📚 Dermatology Encyclopedia",
        "⚕️ FAQ & ABCDE Screening"
    ]
)

st.sidebar.markdown("<hr style='margin: 15px 0;'>", unsafe_allow_html=True)
st.sidebar.markdown("### ⚙️ Preferences")

selected_language = st.sidebar.selectbox(
    "Preferred AI Language:", 
    ["English", "Spanish", "Hindi", "French", "Arabic", "Portuguese"]
)

user_key = st.sidebar.text_input("Gemini API Key (Optional):", type="password", help="Enter key here if not set in .env")
if user_key:
    genai.configure(api_key=user_key)

st.sidebar.markdown("<hr style='margin: 15px 0;'>", unsafe_allow_html=True)
st.sidebar.markdown("""
<div style='font-size: 11px; color: #64748B;'>
    <strong>Disclaimer:</strong> This dashboard is an AI-powered helper tool and is not a replacement for professional dermatological advice.
</div>
""", unsafe_allow_html=True)


# 9. PAGE CONTROLLERS

# PAGE 1: DIAGNOSTIC CENTER
if page == "🏠 Diagnostic Center":
    st.markdown("<h1 class='main-title'>🏠 AI Dermatology Diagnostic Center</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #64748B;'>Upload a photo or capture one using your webcam for CNN prediction and Gemini AI analysis.</p>", unsafe_allow_html=True)
    
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.markdown("<h3 class='section-title'>Input Source</h3>", unsafe_allow_html=True)
        source = st.radio("Choose source:", ["📤 Upload File", "📷 Live Webcam Capture"], horizontal=True)
        
        uploaded_file = None
        if source == "📤 Upload File":
            uploaded_file = st.file_uploader(
                "Upload a skin image",
                type=['jpg', 'jpeg', 'png'],
                help="Accepts JPG, JPEG, and PNG formats."
            )
        else:
            uploaded_file = st.camera_input("Capture skin lesion")
            
    with c2:
        st.markdown("<h3 class='section-title'>Quick Diagnostics</h3>", unsafe_allow_html=True)
        
        if uploaded_file is not None:
            # Manage session state resets on new image
            image_id = f"{uploaded_file.name}_{uploaded_file.size}" if hasattr(uploaded_file, 'name') else f"camera_{uploaded_file.size}"
            if st.session_state.current_image_hash != image_id:
                st.session_state.current_image_hash = image_id
                st.session_state.prediction = ""
                st.session_state.confidence = 0.0
                st.session_state.analysis = ""
                st.session_state.chat_history = []
                
            # Load and display image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Analyzed Specimen", use_column_width=True)
            
            # Predict
            with st.spinner("Classifying image features..."):
                cnn_image = image.resize((224, 224))
                cnn_image = np.array(cnn_image)
                cnn_image = preprocess_input(cnn_image)
                cnn_image = np.expand_dims(cnn_image, axis=0)
                
                prediction = model.predict(cnn_image)
                pred_idx = np.argmax(prediction)
                predicted_class = classes[pred_idx]
                confidence = np.max(prediction) * 100
                
                st.session_state.prediction = predicted_class
                st.session_state.confidence = confidence
                
            # Fetch details
            info = dx_info[predicted_class]
            badge_cls = f"badge-{info['badge']}"
            
            # Display results
            st.markdown(f"""
            <div style='margin-top: 15px;'>
                <p style='font-size: 15px; margin-bottom: 5px; color: #94A3B8;'><strong>Predicted Classification:</strong></p>
                <h2 style='color: #60A5FA; margin-top: 0; margin-bottom: 10px;'>{info['full_name']}</h2>
                <div style='margin-bottom: 15px;'>
                    <span class='{badge_cls}' style='font-size: 14px; padding: 6px 12px;'>{info['type']}</span>
                    <span style='margin-left: 15px; font-weight: 600; color: #E2E8F0;'>Confidence: {confidence:.2f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Progress bar for confidence
            st.progress(float(confidence / 100))
            
            st.markdown(f"""
            <div style='background-color: #1E293B; border-radius: 8px; padding: 15px; border-left: 4px solid #3B82F6; border: 1px solid #334155; margin-top: 15px; color: #E2E8F0;'>
                <p style='margin: 0; color: #E2E8F0;'>🩺 <strong style='color: #F8FAFC;'>Severity Level:</strong> {info['severity']}</p>
                <p style='margin: 5px 0 0 0; color: #E2E8F0;'>📋 <strong style='color: #F8FAFC;'>Dermatologist Recommendation:</strong> {info['action']}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Please supply a skin image to see model classifications.")
            
    # Bottom section for details (Tabs)
    if uploaded_file is not None:
        st.markdown("<hr style='margin: 30px 0;'>", unsafe_allow_html=True)
        tab1, tab2, tab3 = st.tabs([
            "📋 AI Clinical Interpretation", 
            "🔍 Explainable AI (Grad-CAM)", 
            "💬 AI Medical Chatbot"
        ])
        
        # TAB 1: AI CLINICAL REPORT
        with tab1:
            st.subheader("📋 Generative AI Lesion Analysis")
            
            # Run Gemini Vision analysis
            if not st.session_state.analysis:
                with st.spinner(f"Generating clinical interpretation in {selected_language}..."):
                    vision_prompt = f"""
                    Analyze this skin disease image.
                    
                    Please provide your response in the language: {selected_language}.
                    
                    Explain:
                    - possible visual symptoms
                    - color patterns
                    - lesion characteristics
                    - texture
                    - abnormal findings
                    
                    Keep explanation medical but simple, suitable for a patient to understand.
                    Do not make definitive diagnostic claims. Add a disclaimer.
                    """
                    try:
                        response = vision_model.generate_content([
                            vision_prompt,
                            image
                        ])
                        st.session_state.analysis = response.text
                    except Exception as e:
                        st.session_state.analysis = f"Error generating Gemini report: {e}"
                        
            st.markdown(st.session_state.analysis)
            
            # Add Download Report button (PDF format)
            info = dx_info[st.session_state.prediction]
            try:
                # Generate unique case ID based on timestamp
                case_id = f"CASE-{datetime.now().strftime('%Y%m%d%H%M%S')}"
                pdf_data = generate_pdf_report(
                    case_id=case_id,
                    source="📷 Live Webcam Capture" if source == "📷 Live Webcam Capture" else "📤 Uploaded File",
                    language=selected_language,
                    prediction_name=info['full_name'],
                    prediction_code=st.session_state.prediction,
                    confidence=st.session_state.confidence,
                    type_label=info['type'],
                    severity=info['severity'],
                    action=info['action'],
                    analysis=st.session_state.analysis
                )
                
                st.download_button(
                    label="📥 Download Clinical Report (PDF)",
                    data=pdf_data,
                    file_name=f"DermAI_Report_{case_id}.pdf",
                    mime="application/pdf"
                )
            except Exception as e:
                st.error(f"Failed to generate PDF report: {e}")
            
        # TAB 2: EXPLAINABLE AI (GRAD-CAM)
        with tab2:
            st.subheader("🔍 Explainable AI Visual Explanations")
            st.markdown("""
            **Grad-CAM (Gradient-weighted Class Activation Mapping)** highlights the specific visual regions of interest in the image that the deep learning model relied on to make its prediction.
            """)
            
            with st.spinner("Generating Grad-CAM heatmap..."):
                img_array = image.resize((224, 224))
                img_array = np.array(img_array)
                img_array = preprocess_input(img_array)
                img_array = np.expand_dims(img_array, axis=0)
                
                try:
                    heatmap = generate_gradcam(img_array, model)
                    
                    col_gc1, col_gc2 = st.columns(2)
                    with col_gc1:
                        # Slider to control blend transparency
                        alpha_val = st.slider("Heatmap Overlay Transparency (Alpha):", min_value=0.1, max_value=0.9, value=0.45, step=0.05)
                        overlay_img = overlay_gradcam(image, heatmap, alpha=alpha_val)
                        st.image(overlay_img, caption="Grad-CAM Focus Overlay", use_column_width=True)
                    with col_gc2:
                        heatmap_resized = cv2.resize(heatmap, (image.size[0], image.size[1]))
                        heatmap_scaled = np.uint8(255 * heatmap_resized)
                        heatmap_color = cv2.applyColorMap(heatmap_scaled, cv2.COLORMAP_JET)
                        heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
                        st.image(heatmap_color_rgb, caption="Raw Layer Activation Map", use_column_width=True)
                        
                    st.info("💡 **Interpretation:** The bright red areas represent hotspots that the CNN classification layer activated on. This helps clinicians see whether the model focused on the actual lesion or background factors.")
                except Exception as e:
                    st.error(f"Failed to calculate Grad-CAM gradients: {e}")
                    
        # TAB 3: CHATBOT
        with tab3:
            st.subheader("💬 AI Dermatology Consultant")
            st.info(f"Ask the AI questions about the diagnosis, symptoms, or precautions in {selected_language}.")
            
            # Show chat messages
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    
            # Chat input
            if user_question := st.chat_input("Ask a question (e.g., 'Is this contagious?', 'Why did the model predict this?')"):
                # Display user message
                with st.chat_message("user"):
                    st.markdown(user_question)
                st.session_state.chat_history.append({"role": "user", "content": user_question})
                
                # Build context
                chat_history_str = ""
                for msg in st.session_state.chat_history[:-1]:
                    chat_history_str += f"{msg['role'].capitalize()}: {msg['content']}\n"
                    
                full_prompt = f"""
                You are an AI medical assistant. Answer the user's question in the language: {selected_language}.
                
                Context information:
                - CNN prediction: {dx_info[st.session_state.prediction]['full_name']} ({st.session_state.prediction})
                - Confidence score: {st.session_state.confidence:.2f}%
                - Gemini visual analysis of image: {st.session_state.analysis}
                
                Current Chat History:
                {chat_history_str}
                
                User Question:
                {user_question}
                
                Provide a medically relevant, friendly explanation in {selected_language}. Explain symptoms, precautions, and recommend consulting a doctor when appropriate.
                Do not make definitive diagnostic claims, and include a clear disclaimer that you are an AI, not a doctor.
                """
                
                with st.spinner("AI is thinking..."):
                    try:
                        final_response = vision_model.generate_content(full_prompt)
                        ai_reply = final_response.text
                    except Exception as e:
                        ai_reply = f"Error communicating with AI: {e}"
                        
                # Display AI response
                with st.chat_message("assistant"):
                    st.markdown(ai_reply)
                st.session_state.chat_history.append({"role": "assistant", "content": ai_reply})


# PAGE 2: PATIENT ANALYTICS
elif page == "📊 Patient Analytics Dashboard":
    st.markdown("<h1 class='main-title'>📊 Patient Analytics Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #64748B;'>Insights derived from the HAM10000 Skin Lesion Dataset (10,000+ clinical cases)</p>", unsafe_allow_html=True)
    
    df_meta = load_metadata()
    if df_meta is not None:
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("""
            <div class='metric-card'>
                <div class='metric-title'>Total Dataset Records</div>
                <div class='metric-value'>10,015</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>Average Patient Age</div>
                <div class='metric-value'>{df_meta['age'].mean():.1f} yrs</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>Most Prevalent Gender</div>
                <div class='metric-value'>{df_meta['sex'].mode()[0].title()}</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class='metric-card' style='border-left: 5px solid #16A34A;'>
                <div class='metric-title'>Most Common Lesion</div>
                <div class='metric-value'>Common Mole</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Layout: 2 Columns of Charts
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<h3 class='section-title'>Condition Prevalence</h3>", unsafe_allow_html=True)
            counts = df_meta['disease_name'].value_counts().reset_index()
            counts.columns = ['Condition', 'Count']
            fig1 = px.bar(counts, x='Count', y='Condition', orientation='h', color='Count',
                          color_continuous_scale='Blues', labels={'Condition': 'Skin Condition'})
            fig1.update_layout(yaxis={'categoryorder': 'total ascending'}, margin=dict(l=0, r=0, t=10, b=10))
            st.plotly_chart(fig1, use_container_width=True)
            
        with c2:
            st.markdown("<h3 class='section-title'>Lesion Location Distribution</h3>", unsafe_allow_html=True)
            loc_counts = df_meta['localization'].value_counts().reset_index()
            loc_counts.columns = ['Anatomical Location', 'Count']
            fig2 = px.bar(loc_counts, x='Anatomical Location', y='Count', color='Count',
                          color_continuous_scale='Purples')
            fig2.update_layout(xaxis_tickangle=-45, margin=dict(l=0, r=0, t=10, b=10))
            st.plotly_chart(fig2, use_container_width=True)
            
        c3, c4 = st.columns(2)
        with c3:
            st.markdown("<h3 class='section-title'>Age Distribution by Condition</h3>", unsafe_allow_html=True)
            fig3 = px.box(df_meta, x='disease_name', y='age', color='disease_name',
                          labels={'disease_name': 'Condition', 'age': 'Age (Years)'},
                          color_discrete_sequence=px.colors.qualitative.Safe)
            fig3.update_layout(showlegend=False, xaxis_tickangle=-30, margin=dict(l=0, r=0, t=10, b=10))
            st.plotly_chart(fig3, use_container_width=True)
            
        with c4:
            st.markdown("<h3 class='section-title'>Gender Representation</h3>", unsafe_allow_html=True)
            gender_counts = df_meta['sex'].value_counts().reset_index()
            gender_counts.columns = ['Gender', 'Count']
            fig4 = px.pie(gender_counts, values='Count', names='Gender', hole=0.4,
                          color_discrete_sequence=['#3B82F6', '#EC4899', '#94A3B8'])
            fig4.update_layout(margin=dict(l=0, r=0, t=10, b=10))
            st.plotly_chart(fig4, use_container_width=True)
    else:
        st.warning("Metadata file `dataset/HAM10000_metadata.csv` could not be loaded. Please ensure it is present in the repository.")


# PAGE 3: ENCYCLOPEDIA
elif page == "📚 Dermatology Encyclopedia":
    st.markdown("<h1 class='main-title'>📚 Dermatology Encyclopedia</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #64748B;'>A comprehensive dictionary detailing the 7 classifications of skin lesions recognized by our AI system.</p>", unsafe_allow_html=True)
    
    search_q = st.text_input("🔍 Search conditions by name, type, or severity...", "").lower()
    
    for code, info in dx_info.items():
        # Filtering logic
        if search_q and (search_q not in info['full_name'].lower() and search_q not in info['type'].lower() and search_q not in info['severity'].lower()):
            continue
            
        badge_cls = f"badge-{info['badge']}"
        st.markdown(f"""
        <div class='encyclopedia-card'>
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap;'>
                <h3 style='margin: 0; color: #60A5FA;'>{info['full_name']} <span style='font-size: 13px; color: #94A3B8; font-weight: normal;'>({code})</span></h3>
                <span class='{badge_cls}'>{info['type']} — Severity: {info['severity']}</span>
            </div>
            <p style='margin: 4px 0; color: #E2E8F0;'><strong style='color: #F8FAFC;'>Description:</strong> {info['description']}</p>
            <p style='margin: 4px 0; color: #E2E8F0;'><strong style='color: #F8FAFC;'>Common Symptoms:</strong> {info['symptoms']}</p>
            <p style='margin: 4px 0; color: #E2E8F0;'><strong style='color: #F8FAFC;'>Self-Care & Precautions:</strong> {info['precautions']}</p>
            <p style='color: #60A5FA; font-weight: 500; margin: 8px 0 0 0;'>🩺 <strong style='color: #93C5FD;'>Recommended Action:</strong> {info['action']}</p>
        </div>
        """, unsafe_allow_html=True)


# PAGE 4: FAQ & SCREENING
elif page == "⚕️ FAQ & ABCDE Screening":
    st.markdown("<h1 class='main-title'>⚕️ FAQ & ABCDE Screening</h1>", unsafe_allow_html=True)
    
    col_abcde, col_faq = st.columns([3, 2])
    
    with col_abcde:
        st.markdown("<h3 class='section-title'>The ABCDEs of Melanoma Detection</h3>", unsafe_allow_html=True)
        st.markdown("""
        Dermatologists use the **ABCDE method** to identify warning signs of melanoma. Use this framework to guide your monthly self-screenings:
        """)
        
        with st.expander("🅰️ A - ASYMMETRY"):
            st.markdown("**Does one half of the mole match the other half?**  \nBenign moles are usually symmetrical. Melanomas are frequently asymmetrical; if you draw a line through the middle, the two halves will not match.")
        
        with st.expander("🅱️ B - BORDER"):
            st.markdown("**Are the edges irregular, notched, or blurred?**  \nCommon moles have smooth, even borders. Cancerous lesions often have jagged, uneven, or poorly defined edges.")
            
        with st.expander("🅲 C - COLOR"):
            st.markdown("**Does the mole have multiple shades or colors?**  \nBenign moles are typically a single shade of brown. Melanomas can contain different shades of brown, black, red, white, or even blue.")
            
        with st.expander("🅳 D - DIAMETER"):
            st.markdown("**Is the spot larger than 6 millimeters?**  \nMelanomas are usually larger than 6mm (about the size of a pencil eraser) when diagnosed, though they can sometimes be smaller.")
            
        with st.expander("🅴 E - EVOLVING"):
            st.markdown("**Has the mole changed in size, shape, color, or behavior?**  \nAny change in a mole's size, shape, color, or symptoms (such as itching, bleeding, or oozing) is a major warning sign.")
            
        st.warning("⚠️ **Important:** Self-screening is a tool for monitoring, not a formal diagnosis. If you note any of the ABCDE signs, seek immediate evaluation from a board-certified dermatologist.")
        
    with col_faq:
        st.markdown("<h3 class='section-title'>Common Skin Health Questions</h3>", unsafe_allow_html=True)
        
        with st.expander("What causes skin cancer?"):
            st.markdown("Most skin cancers are caused by excessive exposure to ultraviolet (UV) radiation from the sun or tanning beds. UV rays damage skin cell DNA, leading to mutations.")
            
        with st.expander("How often should I check my skin?"):
            st.markdown("Dermatologists recommend performing a full-body self-examination once a month to get familiar with your moles and detect any changes early.")
            
        with st.expander("What SPF should I use daily?"):
            st.markdown("Use a broad-spectrum sunscreen with an SPF of at least 30 every day, even when it is cloudy. Apply it 15 minutes before going outside and reapply every 2 hours.")
            
        with st.expander("Is a model prediction 100% accurate?"):
            st.markdown("No. Artificial Intelligence models are trained on specific datasets (like HAM10000) and have statistical error rates. Predictions are purely educational and cannot replace clinical biopsies or diagnoses.")

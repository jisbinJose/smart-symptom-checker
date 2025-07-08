# Enhanced Eye Disease Detection System with Comprehensive Validations
# Optimized with better validation logic and auto-capture functionality

import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datasets import load_dataset
import os
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import mediapipe as mp
import time

# Set page configuration
st.set_page_config(
    page_title="AI Eye Disease Detection", 
    page_icon="👁️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Styling (Optimized)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Poppins', sans-serif; }
    .main { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    
    .hero-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem; border-radius: 20px; margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1); text-align: center; color: white;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px);
        border-radius: 20px; border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 2rem; margin: 1rem 0; box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .success-card { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
    .warning-card { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
    .error-card { background: linear-gradient(135deg, #fc466b 0%, #3f5efb 100%); }
    
    .success-card, .warning-card, .error-card {
        border-radius: 15px; padding: 1.5rem; margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1); color: white;
    }
    
    .validation-status {
        position: fixed; top: 10px; right: 10px; z-index: 1000;
        padding: 10px 20px; border-radius: 10px; font-weight: bold;
        min-width: 200px; text-align: center;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; border: none; border-radius: 25px; padding: 0.75rem 2rem;
        font-weight: 600; transition: all 0.3s ease;
    }
    
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} .stDeployButton {display:none;}
    </style>
""", unsafe_allow_html=True)

# Disease classes and colors
DISEASE_CLASSES = {
    0: 'Normal', 1: 'Diabetic Retinopathy', 2: 'Glaucoma', 3: 'Cataract',
    4: 'Age-related Macular Degeneration', 5: 'Hypertensive Retinopathy',
    6: 'Pathological Myopia', 7: 'Other'
}

DISEASE_COLORS = {
    'Normal': '#11998e', 'Diabetic Retinopathy': '#fc466b', 'Glaucoma': '#3f5efb',
    'Cataract': '#f093fb', 'Age-related Macular Degeneration': '#ff6b6b',
    'Hypertensive Retinopathy': '#ee5a6f', 'Pathological Myopia': '#667eea', 'Other': '#764ba2'
}

# Enhanced Face and Eye Validator with comprehensive checks
class EnhancedFaceEyeValidator(VideoTransformerBase):
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.7)
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.7, min_tracking_confidence=0.5)
        
        # Validation states
        self.validation_state = {
            'face_count': 0,
            'eyes_open': False,
            'image_quality': False,
            'lighting_good': False,
            'spectacles_detected': False,
            'face_centered': False,
            'stable_detection': False
        }
        
        self.status_message = "Initializing camera..."
        self.access_granted = False
        self.captured_frame = None
        self.stable_frames = 0
        self.required_stable_frames = 10
        self.auto_capture_countdown = 0
        self.max_countdown = 3
        
        # Eye landmarks for validation
        self.LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

    def check_image_quality(self, frame):
        """Check if image quality is sufficient"""
        # Convert to grayscale for quality analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Check for blur using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_threshold = 100
        
        # Check brightness
        brightness = np.mean(gray)
        brightness_ok = 50 < brightness < 200
        
        # Check contrast
        contrast = gray.std()
        contrast_ok = contrast > 30
        
        return laplacian_var > blur_threshold and brightness_ok and contrast_ok

    def check_lighting(self, frame):
        """Check if lighting conditions are adequate"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Check for overexposure and underexposure
        overexposed = np.sum(gray > 240) / gray.size
        underexposed = np.sum(gray < 20) / gray.size
        
        return overexposed < 0.1 and underexposed < 0.1

    def detect_spectacles(self, face_landmarks, frame_shape):
        """Detect if person is wearing spectacles"""
        h, w = frame_shape[:2]
        
        # Get eye region landmarks
        left_eye_points = [(int(face_landmarks.landmark[i].x * w), 
                           int(face_landmarks.landmark[i].y * h)) for i in self.LEFT_EYE]
        right_eye_points = [(int(face_landmarks.landmark[i].x * w), 
                            int(face_landmarks.landmark[i].y * h)) for i in self.RIGHT_EYE]
        
        # Create masks for eye regions
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(left_eye_points)], 255)
        cv2.fillPoly(mask, [np.array(right_eye_points)], 255)
        
        # Convert frame to grayscale and apply edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Check for strong edges around eye regions (indicating spectacle frames)
        eye_region_edges = cv2.bitwise_and(edges, mask)
        edge_density = np.sum(eye_region_edges > 0) / np.sum(mask > 0)
        
        return edge_density > 0.1  # Threshold for spectacle detection

    def calculate_eye_aspect_ratio(self, eye_points):
        """Calculate Eye Aspect Ratio (EAR) to determine if eyes are open"""
        # Vertical eye landmarks
        A = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
        B = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
        
        # Horizontal eye landmark
        C = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
        
        # Calculate EAR
        ear = (A + B) / (2.0 * C)
        return ear

    def check_eyes_open(self, face_landmarks, frame_shape):
        """Check if both eyes are open"""
        h, w = frame_shape[:2]
        
        # Get key eye points for EAR calculation
        left_eye_key = [(int(face_landmarks.landmark[i].x * w), 
                        int(face_landmarks.landmark[i].y * h)) for i in [33, 159, 158, 133, 153, 144]]
        right_eye_key = [(int(face_landmarks.landmark[i].x * w), 
                         int(face_landmarks.landmark[i].y * h)) for i in [362, 385, 386, 263, 373, 380]]
        
        left_ear = self.calculate_eye_aspect_ratio(left_eye_key)
        right_ear = self.calculate_eye_aspect_ratio(right_eye_key)
        
        # Eye is considered open if EAR > 0.25
        return left_ear > 0.25 and right_ear > 0.25

    def check_face_centered(self, detection, frame_shape):
        """Check if face is reasonably centered in frame"""
        h, w = frame_shape[:2]
        bbox = detection.location_data.relative_bounding_box
        
        face_center_x = bbox.xmin + bbox.width / 2
        face_center_y = bbox.ymin + bbox.height / 2
        
        # Check if face center is within middle 60% of frame
        center_threshold = 0.2
        return (center_threshold < face_center_x < (1 - center_threshold) and 
                center_threshold < face_center_y < (1 - center_threshold))

    def draw_validation_overlay(self, frame):
        """Draw validation status overlay on frame"""
        overlay = frame.copy()
        
        # Status indicators
        y_offset = 30
        for key, value in self.validation_state.items():
            color = (0, 255, 0) if value else (0, 0, 255)
            status_text = f"{key.replace('_', ' ').title()}: {'✓' if value else '✗'}"
            cv2.putText(overlay, status_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 25
        
        # Auto-capture countdown
        if self.auto_capture_countdown > 0:
            countdown_text = f"Auto-capture in: {self.auto_capture_countdown}"
            cv2.putText(overlay, countdown_text, (frame.shape[1] - 250, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Status message
        cv2.putText(overlay, self.status_message, (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return overlay

    def transform(self, frame):
        """Main transformation method with comprehensive validation"""
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        
        # Reset validation state
        self.validation_state = {k: False for k in self.validation_state}
        
        # Convert to RGB for MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 1. Face Detection
        face_results = self.mp_face_detection.process(img_rgb)
        
        if not face_results.detections:
            self.status_message = "❌ No face detected"
            self.stable_frames = 0
            return self.draw_validation_overlay(img)
        
        if len(face_results.detections) > 1:
            self.status_message = "❌ Multiple faces detected"
            self.stable_frames = 0
            return self.draw_validation_overlay(img)
        
        self.validation_state['face_count'] = True
        detection = face_results.detections[0]
        
        # 2. Face Mesh for detailed analysis
        mesh_results = self.mp_face_mesh.process(img_rgb)
        
        if not mesh_results.multi_face_landmarks:
            self.status_message = "❌ Face landmarks not detected"
            self.stable_frames = 0
            return self.draw_validation_overlay(img)
        
        face_landmarks = mesh_results.multi_face_landmarks[0]
        
        # 3. Comprehensive Validations
        self.validation_state['image_quality'] = self.check_image_quality(img)
        self.validation_state['lighting_good'] = self.check_lighting(img)
        self.validation_state['eyes_open'] = self.check_eyes_open(face_landmarks, img.shape)
        self.validation_state['face_centered'] = self.check_face_centered(detection, img.shape)
        self.validation_state['spectacles_detected'] = self.detect_spectacles(face_landmarks, img.shape)
        
        # 4. Generate status message based on validations
        if not self.validation_state['image_quality']:
            self.status_message = "❌ Image quality too low (move closer or improve lighting)"
        elif not self.validation_state['lighting_good']:
            self.status_message = "❌ Poor lighting conditions"
        elif not self.validation_state['eyes_open']:
            self.status_message = "❌ Please keep both eyes open"
        elif not self.validation_state['face_centered']:
            self.status_message = "❌ Please center your face in the frame"
        elif self.validation_state['spectacles_detected']:
            self.status_message = "⚠️ Spectacles detected - may affect accuracy"
        else:
            # All validations passed
            self.stable_frames += 1
            self.validation_state['stable_detection'] = True
            
            if self.stable_frames >= self.required_stable_frames:
                # Start auto-capture countdown
                self.auto_capture_countdown = max(0, self.max_countdown - (self.stable_frames - self.required_stable_frames))
                
                if self.auto_capture_countdown > 0:
                    self.status_message = f"✅ All validations passed! Auto-capture in {self.auto_capture_countdown}s"
                else:
                    self.status_message = "✅ Capturing image for analysis..."
                    self.access_granted = True
                    self.captured_frame = img.copy()
                    
                    # Trigger auto-stop
                    if 'auto_capture_triggered' not in st.session_state:
                        st.session_state['auto_capture_triggered'] = True
                        st.session_state['captured_frame'] = img.copy()
            else:
                self.status_message = f"✅ Hold steady... {self.stable_frames}/{self.required_stable_frames}"
        
        # Reset stable frames if any validation fails
        if not all([self.validation_state['face_count'], 
                   self.validation_state['image_quality'],
                   self.validation_state['lighting_good'],
                   self.validation_state['eyes_open'],
                   self.validation_state['face_centered']]):
            self.stable_frames = 0
        
        return self.draw_validation_overlay(img)

# Model and preprocessing functions (optimized)
class EyeDiseaseModel(nn.Module):
    def __init__(self, num_classes=8):
        super(EyeDiseaseModel, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

@st.cache_resource
def load_model():
    model = EyeDiseaseModel(num_classes=len(DISEASE_CLASSES))
    model.eval()
    return model

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return transform(image).unsqueeze(0)

def predict_disease(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return predicted_class, confidence, probabilities[0].numpy()

def create_prediction_plot(probabilities, classes):
    colors = [DISEASE_COLORS.get(disease, '#667eea') for disease in classes.values()]
    
    fig = go.Figure(data=[go.Bar(
        x=list(classes.values()), y=probabilities, marker_color=colors,
        text=[f'{p:.2%}' for p in probabilities], textposition='auto',
        hovertemplate='<b>%{x}</b><br>Probability: %{y:.2%}<extra></extra>'
    )])
    
    fig.update_layout(
        title="🎯 Disease Prediction Probabilities", height=500,
        xaxis_title="Disease Classes", yaxis_title="Probability",
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#2c3e50'), xaxis=dict(tickangle=45)
    )
    
    return fig

def display_prediction_result(disease_name, confidence):
    if disease_name == 'Normal':
        st.markdown(f"""
            <div class="success-card">
                <h2>✅ Healthy Eyes Detected!</h2>
                <h3>Confidence: {confidence:.2%}</h3>
                <p>Your eyes appear to be in good health.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        card_class = "error-card" if confidence > 0.8 else "warning-card" if confidence > 0.6 else "glass-card"
        risk_level = "High Risk" if confidence > 0.8 else "Moderate Risk" if confidence > 0.6 else "Low Risk"
        
        st.markdown(f"""
            <div class="{card_class}">
                <h2>⚠️ {risk_level}: {disease_name}</h2>
                <h3>Confidence: {confidence:.2%}</h3>
                <p>{'Please consult an ophthalmologist immediately.' if confidence > 0.8 else 
                   'Consider scheduling an eye examination.' if confidence > 0.6 else 
                   'Monitor symptoms and consider follow-up.'}</p>
            </div>
        """, unsafe_allow_html=True)

def main():
    # Hero Section
    st.markdown("""
        <div class="hero-container">
            <h1>👁️ AI Eye Disease Detection</h1>
            <p>🤖 Advanced Machine Learning for Ocular Health Analysis</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    
    # Camera capture interface
    st.markdown("""
        <div class="glass-card">
            <h2 style="color: #2c3e50;">📸 Enhanced Real-Time Analysis</h2>
            <p style="color: #7f8c8d;">System performs comprehensive validations including face detection, 
            eye status, image quality, lighting conditions, and spectacle detection.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Validation requirements
    st.markdown("""
        <div class="glass-card">
            <h3 style="color: #2c3e50;">✅ Validation Requirements</h3>
            <ul>
                <li>🎯 <strong>Single Face:</strong> Exactly one face must be detected</li>
                <li>👁️ <strong>Eyes Open:</strong> Both eyes must be clearly open</li>
                <li>🖼️ <strong>Image Quality:</strong> Sufficient sharpness and clarity</li>
                <li>💡 <strong>Good Lighting:</strong> Adequate illumination without glare</li>
                <li>🎭 <strong>Face Centered:</strong> Face should be centered in frame</li>
                <li>👓 <strong>Spectacle Detection:</strong> System will warn if glasses detected</li>
                <li>⚖️ <strong>Stable Detection:</strong> Maintain position for 3 seconds</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'auto_capture_triggered' not in st.session_state:
        st.session_state['auto_capture_triggered'] = False
    if 'captured_frame' not in st.session_state:
        st.session_state['captured_frame'] = None
    if 'analysis_result' not in st.session_state:
        st.session_state['analysis_result'] = None
    
    # WebRTC streamer
    ctx = webrtc_streamer(
        key="enhanced-face-validation",
        video_transformer_factory=EnhancedFaceEyeValidator,
        media_stream_constraints={"video": True, "audio": False},
        async_transform=True,
    )
    
    # Display validation status
    if ctx.video_transformer:
        validation_state = ctx.video_transformer.validation_state
        
        # Create status display
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Validation Status")
            for key, value in validation_state.items():
                icon = "✅" if value else "❌"
                st.write(f"{icon} {key.replace('_', ' ').title()}")
        
        with col2:
            st.markdown("#### 💬 Current Status")
            st.info(ctx.video_transformer.status_message)
    
    # Auto-capture handling
    if st.session_state['auto_capture_triggered'] and st.session_state['captured_frame'] is not None:
        st.markdown("---")
        st.markdown("### 🎯 Auto-Capture Successful!")
        
        # Display captured frame
        captured_img = cv2.cvtColor(st.session_state['captured_frame'], cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(captured_img)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(pil_img, caption="Captured for Analysis", use_column_width=True)
        
        with col2:
            st.markdown("#### 🤖 AI Analysis")
            if st.button("🔍 Analyze Now", key="analyze_captured"):
                with st.spinner("🤖 Analyzing captured image..."):
                    image_tensor = preprocess_image(pil_img)
                    predicted_class, confidence, probabilities = predict_disease(model, image_tensor)
                    disease_name = DISEASE_CLASSES[predicted_class]
                    
                    # Store results
                    st.session_state['analysis_result'] = {
                        "image": pil_img,
                        "disease_name": disease_name,
                        "confidence": confidence,
                        "probabilities": probabilities
                    }
                    
                    # Display results
                    display_prediction_result(disease_name, confidence)
                    
                    # Show detailed analysis
                    st.markdown("---")
                    st.markdown("### 📊 Detailed Analysis")
                    
                    fig = create_prediction_plot(probabilities, DISEASE_CLASSES)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Probability table
                    prob_df = pd.DataFrame({
                        'Disease': list(DISEASE_CLASSES.values()),
                        'Probability': probabilities,
                        'Percentage': [f"{p:.2%}" for p in probabilities],
                        'Risk Level': ['High' if p > 0.8 else 'Medium' if p > 0.6 else 'Low' for p in probabilities]
                    }).sort_values('Probability', ascending=False)
                    
                    st.dataframe(prob_df, use_container_width=True)
                    
                    # Recommendations
                    st.markdown("### 💡 Recommendations")
                    if confidence > 0.8:
                        st.success("High confidence prediction - Consider professional consultation")
                    elif confidence > 0.6:
                        st.warning("Medium confidence - Monitor and consider follow-up")
                    else:
                        st.info("Low confidence - May need better image quality")
                    
                    # Auto-redirect to results
                    st.success("✅ Analysis complete! Results saved.")
                    time.sleep(2)
                    st.rerun()
    
    # Display saved results if available
    if st.session_state['analysis_result']:
        st.markdown("---")
        st.markdown("### 📄 Recent Analysis Results")
        
        result = st.session_state['analysis_result']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(result["image"], caption="Analyzed Image", use_column_width=True)
        
        with col2:
            st.markdown(f"""
                <div class="glass-card">
                    <h3>📊 Analysis Summary</h3>
                    <p><strong>Prediction:</strong> {result['disease_name']}</p>
                    <p><strong>Confidence:</strong> {result['confidence']:.2%}</p>
                    <p><strong>Status:</strong> Analysis Complete</p>
                </div>
            """, unsafe_allow_html=True)
    
    # Reset button
    if st.button("🔄 Reset Analysis", key="reset_analysis"):
        st.session_state['auto_capture_triggered'] = False
        st.session_state['captured_frame'] = None
        st.session_state['analysis_result'] = None
        st.rerun()

if __name__ == "__main__":
    main()
# Eye Disease Detection System

A real-time and static image-based eye disease prediction system using deep learning and computer vision, with a modern Streamlit web interface.

## Features
- Real-time webcam detection and prediction
- Single image upload and disease prediction
- Batch image folder prediction and report
- Camera settings
- Save results and session summary
- Grad-CAM visualizations for explainability

## Setup
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download or train the model and place `eye_disease_resnet18.pth` in the project root.
4. Run the app:
   ```bash
   streamlit run app.py
   ```

## Dataset
- Uses the [falah/eye-disease-dataset](https://huggingface.co/datasets/Falah/eye-disease-dataset) (requires Hugging Face authentication)

## Usage
- Use the sidebar to select between real-time webcam, single image, or batch prediction modes.
- View predictions, confidence scores, and Grad-CAM visualizations.
- Save results and view session summaries.

## Notes
- For webcam support, ensure your device has a camera and permissions are granted.
- For Grad-CAM, only ResNet18 is currently supported. 
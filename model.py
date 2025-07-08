import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

DISEASE_CLASSES = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']

# Model loading

def load_model(model_path='eye_disease_resnet18.pth', num_classes=4):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Prediction function

def predict_image(model, image: Image.Image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred_idx = probs.argmax()
        return DISEASE_CLASSES[pred_idx], float(probs[pred_idx]), probs 
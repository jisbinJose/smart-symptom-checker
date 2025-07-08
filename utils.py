import torch
import numpy as np
import cv2
from PIL import Image

# Grad-CAM utility

def generate_gradcam(model, image_tensor, target_layer_name='layer4'):
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    def forward_hook(module, input, output):
        activations.append(output.detach())

    # Register hooks
    target_layer = dict([*model.named_modules()])[target_layer_name]
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)

    model.zero_grad()
    output = model(image_tensor)
    class_idx = output.argmax(dim=1).item()
    score = output[0, class_idx]
    score.backward()

    grads = gradients[0][0]
    acts = activations[0][0]
    pooled_grads = torch.mean(grads, dim=[1, 2])
    for i in range(acts.shape[0]):
        acts[i, :, :] *= pooled_grads[i]
    heatmap = acts.mean(dim=0).cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap

# Session log utility

class SessionLog:
    def __init__(self):
        self.entries = []
    def add(self, filename, prediction, confidence, timestamp):
        self.entries.append({
            'filename': filename,
            'prediction': prediction,
            'confidence': confidence,
            'timestamp': timestamp
        })
    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame(self.entries) 
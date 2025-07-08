from PIL import Image, ImageEnhance
import numpy as np

def preprocess_image(image: Image.Image):
    # Resize
    image = image.resize((224, 224))
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)
    # Convert to RGB if not already
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image 
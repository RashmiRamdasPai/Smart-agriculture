import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Load the model
model = tf.keras.models.load_model("backend/models/disease_model.h5")

# Load class labels from directory or manually define
# Example: class_indices = {0: 'Healthy', 1: 'Rust', 2: 'Blight'}
class_indices = {0: 'Bacterial Spot', 1: 'Early Blight', 2: 'Late Blight', 3: 'Leaf Mold', 4: 'Healthy'}

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_disease(image_bytes):
    img_array = preprocess_image(image_bytes)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    class_label = class_indices.get(predicted_class, "Unknown")
    confidence = float(np.max(prediction))
    return {
        "disease": class_label,
        "confidence": round(confidence * 100, 2)
    }
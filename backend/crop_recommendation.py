import pickle
import numpy as np
from pydantic import BaseModel

# Load the trained model
with open("backend/models/crop_model.pkl", "rb") as f:
    crop_model = pickle.load(f)

# Input data model (if needed outside main.py)
class CropInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

def recommend_crop(data: CropInput):
    input_array = np.array([[data.N, data.P, data.K, data.temperature, data.humidity, data.ph, data.rainfall]])
    prediction = crop_model.predict(input_array)[0]
    return {"recommended_crop": prediction}

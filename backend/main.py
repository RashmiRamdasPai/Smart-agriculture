from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import pickle
import json
import random

# For STT & TTS (Kannada voice assistant)
import speech_recognition as sr
from gtts import gTTS
import os

# For image prediction
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==== 1. Voice Chatbot (Kannada) ====
with open("chatbot_model.pkl", "rb") as f:
    chatbot_model = pickle.load(f)

with open("intents_kan.json", encoding="utf-8") as f:
    intents = json.load(f)

def get_intent_tag(user_input):
    pred = chatbot_model.predict([user_input])[0]
    for intent in intents["intents"]:
        if intent["tag"] == pred:
            return random.choice(intent["responses"])
    return "ಕ್ಷಮಿಸಿ, ನನಗೆ ಅರ್ಥವಾಗಲಿಲ್ಲ."

@app.post("/ask_kannada")
async def ask_kannada(text: str):
    response = get_intent_tag(text)
    return {"response": response}

# ==== 2. Plant Disease Detection ====
model = load_model("backend/models/disease_model.h5")  # Use your own or pretrained MobileNet model
with open("backend/models/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

@app.post("/predict_disease")
async def predict_disease(file: UploadFile = File(...)):
    img = Image.open(file.file).resize((224, 224))
    arr = np.expand_dims(np.array(img)/255.0, axis=0)
    pred = model.predict(arr)
    label = label_encoder.inverse_transform([np.argmax(pred)])[0]
    return {"disease": label}

# ==== 3. Crop Recommendation ====
class CropInput(BaseModel):
    nitrogen: float
    phosphorous: float
    potassium: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

with open("crop_model.pkl", "rb") as f:
    crop_model = pickle.load(f)

@app.post("/recommend_crop")
async def recommend_crop(data: CropInput):
    X = [[
        data.nitrogen, data.phosphorous, data.potassium,
        data.temperature, data.humidity, data.ph, data.rainfall
    ]]
    prediction = crop_model.predict(X)[0]
    return {"recommended_crop": prediction}

# ==== 4. Weather Alerts ====
@app.get("/weather_alert")
def get_weather():
    return {
        "location": "Udupi",
        "temperature": 30,
        "condition": "Moderate Rain",
        "advice": "ಹುಲ್ಲು ಕಡಿತ ಮತ್ತು ಪೆಸ್ಟಿಸೈಡ್ ಬಳಕೆ ತಪ್ಪಿಸಿ"
    }

# ==== 5. E-Commerce ====
products = [
    {"id": 1, "name": "Fertilizer A", "price": 300, "quantity": 25},
    {"id": 2, "name": "Pesticide B", "price": 150, "quantity": 10},
    {"id": 3, "name": "Tractor Rent", "price": 2000, "quantity": 5}
]

@app.get("/products")
def list_products():
    return products

@app.post("/buy/{product_id}")
def buy(product_id: int):
    for item in products:
        if item["id"] == product_id:
            if item["quantity"] > 0:
                item["quantity"] -= 1
                return {"message": f"Purchased {item['name']}!"}
            else:
                return {"message": "Out of stock"}
    return {"message": "Product not found"}

# ==== 6. Government Schemes ====
schemes_data = [
    {
        "title": "PM-KISAN Samman Nidhi",
        "description": "Provides ₹6000/year to all landholding farmers.",
        "link": "https://pmkisan.gov.in"
    },
    {
        "title": "Kisan Credit Card (KCC)",
        "description": "Low-interest credit facility for farmers.",
        "link": "https://www.pmkisan.gov.in/Documents/KCC_FAQs.pdf"
    },
    {
        "title": "NABARD Schemes",
        "description": "Funding support for farm infrastructure.",
        "link": "https://www.nabard.org"
    }
]

@app.get("/schemes")
def get_schemes():
    return schemes_data

# Optional: Uncomment for direct running
# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

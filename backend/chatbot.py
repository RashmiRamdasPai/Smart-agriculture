import pickle
import speech_recognition as sr
from gtts import gTTS
import os
import uuid

from backend.voice_chatbot import recognize_kannada_speech, speak_kannada

# Load chatbot model
with open("backend/models/chatbot_model.pkl", "rb") as f:
    model = pickle.load(f)

def ask_kannada_chatbot(audio_file):
    # Save the uploaded audio file temporarily
    temp_audio_path = f"static/uploads/{uuid.uuid4()}.wav"
    with open(temp_audio_path, "wb") as f:
        f.write(audio_file.file.read())

    # Convert speech to text (Kannada)
    try:
        query_text = recognize_kannada_speech(temp_audio_path)
    except Exception as e:
        return {"error": "Could not recognize speech", "details": str(e)}

    # Predict intent using trained model
    response_text = model.predict([query_text])[0]

    # Convert text response to speech (Kannada)
    audio_path = speak_kannada(response_text)

    return {
        "query_text": query_text,
        "response_text": response_text,
        "audio_file": audio_path
    }
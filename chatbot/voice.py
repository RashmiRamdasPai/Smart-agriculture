import speech_recognition as sr
from gtts import gTTS
import uuid
import os

UPLOAD_DIR = "static/audio_responses"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def recognize_kannada_speech(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
    
    # Recognize using Google STT for Kannada
    text = recognizer.recognize_google(audio_data, language='kn-IN')
    return text

def speak_kannada(text):
    tts = gTTS(text=text, lang='kn')
    filename = f"{uuid.uuid4()}.mp3"
    filepath = os.path.join(UPLOAD_DIR, filename)
    tts.save(filepath)
    return filepath
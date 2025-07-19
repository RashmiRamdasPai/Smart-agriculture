import json
import pickle
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

nltk.download('punkt')

# Load intents
with open("chatbot/intents_kan.json", encoding="utf-8") as file:
    data = json.load(file)

# Preprocess data
X = []
y = []
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        X.append(pattern)
        y.append(intent["tag"])

# Build model pipeline
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X, y)

# Save model
with open("backend/models/chatbot_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Kannada chatbot model saved to backend/models/chatbot_model.pkl")

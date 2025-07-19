import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
print("Loading dataset...")
data = pd.read_csv("datasets/crop_recommendation.csv")

# Features and target
X = data.drop("label", axis=1)
y = data["label"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
print("Training Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model
with open("backend/models/crop_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Crop recommendation model saved to backend/models/crop_model.pkl")

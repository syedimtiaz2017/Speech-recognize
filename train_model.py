import os
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from utils.feature import extract_features

print(os.listdir())
DATASET_PATH = "dataset_clean"

X = []
y = []

for emotion in os.listdir(DATASET_PATH):
    emotion_path = os.path.join(DATASET_PATH, emotion)

    for file in os.listdir(emotion_path):
        if file.endswith(".wav"):
            file_path = os.path.join(emotion_path, file)

            features = extract_features(file_path)
            if features is not None:
                X.append(features)
                y.append(emotion)

X = np.array(X)
y = np.array(y)

print("Training samples:", len(X))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=200)
model.fit(X_scaled, y)

# Save
os.makedirs("model", exist_ok=True)

pickle.dump(model, open("model/model.pkl", "wb"))
pickle.dump(scaler, open("model/scaler.pkl", "wb"))

print("✅ Model trained and saved!")

import os
import shutil

DATASET_PATH = "dataset"
OUTPUT_PATH = "dataset_clean"

emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

os.makedirs(OUTPUT_PATH, exist_ok=True)

for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        if file.endswith(".wav"):
            parts = file.split("-")
            emotion_code = parts[2]

            emotion = emotion_map.get(emotion_code)

            if emotion:
                emotion_folder = os.path.join(OUTPUT_PATH, emotion)
                os.makedirs(emotion_folder, exist_ok=True)

                src = os.path.join(root, file)
                dst = os.path.join(emotion_folder, file)

                shutil.copy(src, dst)

print("✅ Dataset organized into emotion folders!")

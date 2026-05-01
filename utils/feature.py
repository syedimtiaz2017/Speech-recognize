import librosa
import numpy as np

def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, duration=3, offset=0.5)

        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        mel = librosa.feature.melspectrogram(y=audio, sr=sr)

        mfcc = np.mean(mfcc.T, axis=0)
        chroma = np.mean(chroma.T, axis=0)
        mel = np.mean(mel.T, axis=0)

        return np.hstack([mfcc, chroma, mel])

    except Exception as e:
        print("Error:", e)
        return None
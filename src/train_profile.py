# src/train_profile.py

import sounddevice as sd
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
import os
from scipy.io.wavfile import write

DURATION_SECONDS = 30
SAMPLE_RATE = 16000
PROFILE_SAVE_PATH = "../models/my_voice_profile.npy"

def record_voice(duration, sample_rate):
    print(f"Recording your voice for {duration} seconds...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    return recording.flatten()

def save_voice_profile(audio, sample_rate, save_path):
    temp_wav = "temp_recording.wav"
    write(temp_wav, sample_rate, audio)
    wav = preprocess_wav(temp_wav)
    encoder = VoiceEncoder()
    embedding = encoder.embed_utterance(wav)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, embedding)
    os.remove(temp_wav)
    print(f"✅ Voice profile saved to {save_path}")

if __name__ == "__main__":
    audio = record_voice(DURATION_SECONDS, SAMPLE_RATE)
    save_voice_profile(audio, SAMPLE_RATE, PROFILE_SAVE_PATH)

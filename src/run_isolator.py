# src/run_isolator.py

import sounddevice as sd
import numpy as np
import queue
import yaml
from resemblyzer import VoiceEncoder, preprocess_wav
from scipy.io.wavfile import write
import os

# Load config
with open("../config.yaml", "r") as f:
    config = yaml.safe_load(f)

INPUT_DEVICE = config["input_device"]
OUTPUT_DEVICE = config["output_device"]
SAMPLE_RATE = config["sample_rate"]
FRAME_DURATION_MS = config["frame_duration_ms"]
MATCH_THRESHOLD = config["match_threshold"]
PROFILE_PATH = "../models/my_voice_profile.npy"

your_embedding = np.load(PROFILE_PATH)
encoder = VoiceEncoder()
q = queue.Queue()

def audio_callback(indata, frames, time, status):
    q.put(indata.copy())

def match_voice(chunk):
    temp_chunk = "temp_chunk.wav"
    write(temp_chunk, SAMPLE_RATE, chunk)
    wav = preprocess_wav(temp_chunk)
    chunk_embedding = encoder.embed_utterance(wav)
    os.remove(temp_chunk)

    similarity = np.dot(your_embedding, chunk_embedding) / (np.linalg.norm(your_embedding) * np.linalg.norm(chunk_embedding))
    return similarity

def main():
    frame_samples = int(SAMPLE_RATE * (FRAME_DURATION_MS / 1000))

    print("🔎 Listening for your voice...")

    with sd.InputStream(samplerate=SAMPLE_RATE, device=INPUT_DEVICE, channels=1, dtype='float32', callback=audio_callback):
        with sd.OutputStream(samplerate=SAMPLE_RATE, device=OUTPUT_DEVICE, channels=1, dtype='float32') as outstream:
            buffer = np.zeros((0,), dtype='float32')
            
            while True:
                while not q.empty():
                    buffer = np.concatenate((buffer, q.get()[:, 0]))
                
                if len(buffer) >= frame_samples:
                    frame = buffer[:frame_samples]
                    buffer = buffer[frame_samples:]

                    sim = match_voice(frame)
                    if sim > MATCH_THRESHOLD:
                        outstream.write(frame.reshape(-1, 1))

if __name__ == "__main__":
    main()

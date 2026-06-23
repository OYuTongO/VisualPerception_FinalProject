"""把 diana_50min.wav 切成 5 分钟小块"""
import os
import numpy as np
import soundfile as sf

INPUT_WAV  = "diana_voice_raw/diana_50min.wav"
CHUNKS_DIR = "diana_voice_raw/chunks"
CHUNK_MIN  = 5

os.makedirs(CHUNKS_DIR, exist_ok=True)
audio, sr = sf.read(INPUT_WAV)
chunk_samples = CHUNK_MIN * 60 * sr
n_chunks = int(np.ceil(len(audio) / chunk_samples))
print(f"Splitting into {n_chunks} chunks...")
for i in range(n_chunks):
    chunk = audio[i*chunk_samples : (i+1)*chunk_samples]
    sf.write(f"{CHUNKS_DIR}/chunk_{i:03d}.wav", chunk, sr)
    print(f"  chunk_{i:03d}.wav: {len(chunk)/sr:.1f}s")
print("Done splitting.")

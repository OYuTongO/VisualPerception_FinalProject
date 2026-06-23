"""把各块 vocals.wav 拼成完整的 diana_50min/vocals.wav"""
import os
import glob
import numpy as np
import soundfile as sf

CHUNKS_SEP_DIR = "diana_voice_raw/chunks/separated/htdemucs"
OUTPUT_PATH    = "diana_voice_raw/separated/htdemucs/diana_50min/vocals.wav"

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

chunk_dirs = sorted(glob.glob(f"{CHUNKS_SEP_DIR}/chunk_*/"))
print(f"Found {len(chunk_dirs)} separated chunks")

parts, sr = [], None
for d in chunk_dirs:
    path = os.path.join(d, "vocals.wav")
    audio, s = sf.read(path)
    if sr is None:
        sr = s
    parts.append(audio)
    print(f"  {path}: {len(audio)/s:.1f}s")

full = np.concatenate(parts, axis=0)
sf.write(OUTPUT_PATH, full, sr)
print(f"\nSaved: {OUTPUT_PATH}  ({len(full)/sr/60:.1f} min)")

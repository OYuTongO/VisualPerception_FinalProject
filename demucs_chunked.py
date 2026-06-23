"""
把长音频切成小块分批跑 demucs，最后拼回一个完整的 vocals.wav
用法: python demucs_chunked.py
"""
import os
import sys
import subprocess
import numpy as np
import soundfile as sf

INPUT_WAV   = "diana_voice_raw/diana_50min.wav"
OUTPUT_DIR  = "diana_voice_raw/separated/htdemucs/diana_50min"
CHUNKS_DIR  = "diana_voice_raw/chunks"
CHUNK_MIN   = 5   # 每块时长（分钟）

os.makedirs(CHUNKS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. 切块
print("Loading audio...")
audio, sr = sf.read(INPUT_WAV)
chunk_samples = CHUNK_MIN * 60 * sr
n_chunks = int(np.ceil(len(audio) / chunk_samples))
print(f"Total {len(audio)/sr/60:.1f} min → {n_chunks} chunks of {CHUNK_MIN} min each")

chunk_paths = []
for i in range(n_chunks):
    chunk = audio[i*chunk_samples : (i+1)*chunk_samples]
    path = os.path.join(CHUNKS_DIR, f"chunk_{i:03d}.wav")
    sf.write(path, chunk, sr)
    chunk_paths.append(path)
    print(f"  Written chunk {i:03d}: {len(chunk)/sr:.1f}s")

# 2. 逐块跑 demucs
vocals_chunks = []
SKIP_DONE = True  # chunk_000 已处理完，跳过
for i, path in enumerate(chunk_paths):
    if SKIP_DONE and i == 0:
        print(f"[1/{n_chunks}] chunk_000 already done, skipping.")
        continue
    print(f"\n[{i+1}/{n_chunks}] Running demucs on {os.path.basename(path)} ...")
    result = subprocess.run(
        [sys.executable, "-m", "demucs", "--two-stems", "vocals",
         "-o", CHUNKS_DIR + "/separated", path],
    )
    if result.returncode != 0:
        print("demucs failed on", path)
        sys.exit(1)

    vocals_path = os.path.join(
        CHUNKS_DIR, "separated", "htdemucs",
        os.path.splitext(os.path.basename(path))[0],
        "vocals.wav"
    )
    v, vsr = sf.read(vocals_path)
    vocals_chunks.append(v)
    print(f"  → vocals {len(v)/vsr:.1f}s loaded")

# 3. 拼回完整 vocals.wav
print("\nConcatenating vocals...")
full_vocals = np.concatenate(vocals_chunks, axis=0)
out_path = os.path.join(OUTPUT_DIR, "vocals.wav")
sf.write(out_path, full_vocals, vsr)
print(f"Done! Saved to {out_path}  ({len(full_vocals)/vsr/60:.1f} min)")

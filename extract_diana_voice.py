"""
用 pyannote-audio 做说话人分离，提取Diana的所有语音片段
用法：
  Step 1: python extract_diana_voice.py --diarize   # 分析说话人，打印各段时间轴
  Step 2: 听一下输出，确认哪个 SPEAKER_XX 是Diana
  Step 3: python extract_diana_voice.py --extract --speaker SPEAKER_01  # 提取Diana的片段
"""
# huggingface_hub 1.x removed use_auth_token; patch before pyannote imports it
import functools
import huggingface_hub as _hfhub

def _compat_wrap(fn):
    @functools.wraps(fn)
    def wrapper(*args, use_auth_token=None, **kwargs):
        if use_auth_token is not None and "token" not in kwargs:
            kwargs["token"] = use_auth_token
        return fn(*args, **kwargs)
    return wrapper

_hfhub.hf_hub_download = _compat_wrap(_hfhub.hf_hub_download)
if hasattr(_hfhub, "snapshot_download"):
    _hfhub.snapshot_download = _compat_wrap(_hfhub.snapshot_download)

import argparse
import os
import torch
import numpy as np
import soundfile as sf
from pyannote.audio import Pipeline

# ⚠️ 不要把这个token提交到git
HF_TOKEN = "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"

INPUT_WAV = "diana_voice_raw/separated/htdemucs/diana_50min/vocals.wav"
OUTPUT_DIR = "diana_voice_raw/diana_segments"
RTTM_FILE  = "diana_voice_raw/diarization.rttm"

MIN_DURATION = 2.0   # 最短保留片段(秒)，太短的句子丢掉
MAX_DURATION = 15.0  # 最长片段(秒)，超过的截断


def run_diarization():
    print("Loading pyannote pipeline ...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HF_TOKEN
    )
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
        print("Using GPU")

    # 用soundfile预加载，绕过torchcodec/FFmpeg依赖
    print(f"Loading audio from {INPUT_WAV} ...")
    waveform, sample_rate = sf.read(INPUT_WAV)
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]   # (1, T)
    else:
        waveform = waveform.T                 # (C, T)
    audio_input = {
        "waveform": torch.from_numpy(waveform).float(),
        "sample_rate": sample_rate
    }

    print(f"Running diarization ...")
    diarization = pipeline(audio_input)

    # 保存RTTM文件供后续使用
    with open(RTTM_FILE, "w") as f:
        diarization.write_rttm(f)
    print(f"Saved diarization to {RTTM_FILE}")

    # 统计每个说话人的总时长
    speaker_duration = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        dur = turn.end - turn.start
        speaker_duration[speaker] = speaker_duration.get(speaker, 0) + dur

    print("\n===== 说话人统计 =====")
    for spk, dur in sorted(speaker_duration.items(), key=lambda x: -x[1]):
        print(f"  {spk}: {dur/60:.1f} 分钟 ({int(dur)} 秒)")

    print("\n===== 前30条片段（听一下判断哪个是Diana）=====")
    count = 0
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if count >= 30:
            break
        print(f"  {turn.start:6.1f}s - {turn.end:6.1f}s  [{speaker}]  ({turn.end-turn.start:.1f}s)")
        count += 1

    print(f"\n确认Diana是哪个说话人后，运行：")
    print(f"  python extract_diana_voice.py --extract --speaker SPEAKER_XX")


def extract_speaker(target_speaker):
    if not os.path.exists(RTTM_FILE):
        print("请先运行 --diarize")
        return

    from pyannote.core import Annotation
    with open(RTTM_FILE, "r") as f:
        lines = f.readlines()

    # 解析RTTM
    segments = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 8:
            continue
        start = float(parts[3])
        dur   = float(parts[4])
        spk   = parts[7]
        if spk == target_speaker and dur >= MIN_DURATION:
            segments.append((start, start + min(dur, MAX_DURATION)))

    print(f"找到 {len(segments)} 个 {target_speaker} 的片段")

    # 读取音频
    audio, sr = sf.read(INPUT_WAV)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # 转单声道

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    saved = 0
    for i, (start, end) in enumerate(segments):
        s = int(start * sr)
        e = int(end * sr)
        chunk = audio[s:e]
        out_path = os.path.join(OUTPUT_DIR, f"diana_{i:04d}.wav")
        sf.write(out_path, chunk, sr)
        saved += 1

    print(f"已保存 {saved} 个片段到 {OUTPUT_DIR}/")
    print(f"总时长: {sum(e-s for s,e in segments):.0f} 秒")

    # 生成list文件（供GPT-SoVITS训练用，文本留空后续ASR填充）
    list_path = "diana_voice_raw/diana_segments.list"
    with open(list_path, "w", encoding="utf-8") as f:
        for i in range(saved):
            wav_path = os.path.abspath(
                os.path.join(OUTPUT_DIR, f"diana_{i:04d}.wav")
            ).replace("\\", "/")
            f.write(f"{wav_path}|diana|EN|\n")
    print(f"生成列表文件: {list_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--diarize",  action="store_true", help="分析说话人")
    parser.add_argument("--extract",  action="store_true", help="提取指定说话人")
    parser.add_argument("--speaker",  type=str, default="SPEAKER_01", help="目标说话人ID")
    args = parser.parse_args()

    if args.diarize:
        run_diarization()
    elif args.extract:
        extract_speaker(args.speaker)
    else:
        parser.print_help()

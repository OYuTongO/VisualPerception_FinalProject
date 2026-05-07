# ASL Gesture Recognition Learning Game

A real-time American Sign Language (ASL) alphabet learning game powered by computer vision.

## Requirements

- Python 3.10+
- Webcam

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

Download the ASL Alphabet dataset from Kaggle:

```bash
# 1. Place your kaggle.json in ~/.kaggle/
# 2. Run:
kaggle datasets download grassknoted/asl-alphabet
unzip asl-alphabet.zip -d data/raw/
```

## Run

```bash
python src/game/main.py
```

## Development Progress

See [PLAN.md](PLAN.md) for the full development roadmap and current progress.

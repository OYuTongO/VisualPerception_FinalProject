# ASL Gesture Recognition Learning Game

A real-time American Sign Language (ASL) alphabet learning game powered by computer vision.

## Requirements

- Python 3.10+
- Webcam
- `model/hand_landmarker.task`
- `model/asl_classifier.pkl`
- `game_module/D.jpg` for Diana's avatar
- `game_module/X.jpg` for the learner avatar

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

Download the ASL Alphabet dataset from Kaggle if you need to rebuild `data/landmarks.csv`:

```bash
# 1. Place your kaggle.json in ~/.kaggle/
# 2. Run:
kaggle datasets download grassknoted/asl-alphabet
unzip asl-alphabet.zip -d data/raw/
```

## Build the classifier

If `model/asl_classifier.pkl` is missing, train it from the included landmarks CSV:

```bash
python src/train_model.py
```

`model/hand_landmarker.task` must be downloaded separately from the MediaPipe Hand Landmarker model page and placed under `model/`.

## Run the game

From the project root:

```bash
python -m src.game.main
```

Controls:

- Sign the highlighted ASL letter in front of the webcam.
- Correct letters enlarge on screen, glow, fly into the learner reply slot, then turn gold.
- The learner avatar mirrors the current recognized letter with a gesture card.
- Diana shows the current target gesture and demonstrates the correct letter after mistakes.
- Wrong letters show a red feedback state, increase the error counter, and pause input until Diana's demo ends.
- Completing each reply shows a green success state.
- Press `Q` or `ESC` to quit.

## Development Progress

See [PLAN.md](PLAN.md) for the full development roadmap and current progress.

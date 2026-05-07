import os
import csv
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

DATA_DIR = os.path.join("data", "raw", "asl_alphabet_train", "asl_alphabet_train")
OUTPUT_CSV = os.path.join("data", "landmarks.csv")
MODEL_PATH = os.path.join("model", "hand_landmarker.task")

SKIP_LABELS = {"J", "Z", "del", "nothing", "space"}

def build_landmarker():
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.3,
    )
    return mp_vision.HandLandmarker.create_from_options(options)

def normalize_landmarks(landmarks):
    """归一化到手部边界框，消除位置和尺度影响。"""
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    w = x_max - x_min or 1e-6
    h = y_max - y_min or 1e-6
    result = []
    for lm in landmarks:
        result.append((lm.x - x_min) / w)
        result.append((lm.y - y_min) / h)
    return result

def main():
    labels = sorted([
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d)) and d not in SKIP_LABELS
    ])
    print(f"处理类别（{len(labels)}个）：{labels}\n")

    header = ["label"] + [coord
                          for i in range(21)
                          for coord in (f"x{i}", f"y{i}")]

    total_ok = total_skip = 0

    with build_landmarker() as landmarker, open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for label in labels:
            folder = os.path.join(DATA_DIR, label)
            files = [fn for fn in os.listdir(folder)
                     if fn.lower().endswith((".jpg", ".jpeg", ".png"))]
            ok = skip = 0

            for fname in files:
                img_path = os.path.join(folder, fname)
                img = cv2.imread(img_path)
                if img is None:
                    skip += 1
                    continue

                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect(mp_image)

                if not result.hand_landmarks:
                    skip += 1
                    continue

                coords = normalize_landmarks(result.hand_landmarks[0])
                writer.writerow([label] + coords)
                ok += 1

            print(f"  {label}: {ok} 成功 / {skip} 跳过")
            total_ok += ok
            total_skip += skip

    print(f"\n完成！共 {total_ok} 条写入 {OUTPUT_CSV}，{total_skip} 张图跳过（未检测到手）")

if __name__ == "__main__":
    main()

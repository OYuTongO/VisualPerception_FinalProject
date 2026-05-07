import os
import pickle
import collections
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH      = os.path.join("model", "asl_classifier.pkl")
LANDMARKER_PATH = os.path.join("model", "hand_landmarker.task")
CONFIDENCE_THRESHOLD = 0.6
SMOOTH_FRAMES        = 5   # 连续N帧一致才确认结果


class Recognizer:
    def __init__(self, model_path=MODEL_PATH, landmarker_path=LANDMARKER_PATH):
        with open(model_path, "rb") as f:
            self._clf = pickle.load(f)

        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=landmarker_path),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.5,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._raw_landmarks = None          # 原始 landmark 对象列表，供外部绘制骨架
        self._smooth_buf = collections.deque(maxlen=SMOOTH_FRAMES)

    def predict(self, frame: np.ndarray):
        """
        输入 BGR numpy 图像，返回 (letter, confidence)。
        未检测到手或置信度不足时返回 (None, 0.0)。
        连续 SMOOTH_FRAMES 帧一致才输出结果，避免抖动。
        """
        self._raw_landmarks = None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_img)

        if not result.hand_landmarks:
            self._smooth_buf.clear()
            return None, 0.0

        self._raw_landmarks = result.hand_landmarks[0]
        coords = self._normalize(self._raw_landmarks)

        proba = self._clf.predict_proba([coords])[0]
        confidence = float(proba.max())
        letter = self._clf.classes_[proba.argmax()]

        if confidence < CONFIDENCE_THRESHOLD:
            self._smooth_buf.clear()
            return None, confidence

        self._smooth_buf.append(letter)
        # 只有缓冲区满且全部一致时才输出
        if (len(self._smooth_buf) == SMOOTH_FRAMES
                and len(set(self._smooth_buf)) == 1):
            return letter, confidence

        return None, confidence

    def get_landmarks(self):
        """返回原始 landmark 列表（mediapipe NormalizedLandmark），供 Pygame 绘制骨架。"""
        return self._raw_landmarks

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    @staticmethod
    def _normalize(landmarks):
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        w = xmax - xmin or 1e-6
        h = ymax - ymin or 1e-6
        result = []
        for lm in landmarks:
            result.append((lm.x - xmin) / w)
            result.append((lm.y - ymin) / h)
        return result


# MediaPipe 手部骨架连接关系（用于绘制骨架线）
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # 拇指
    (0, 5), (5, 6), (6, 7), (7, 8),        # 食指
    (0, 9), (9, 10), (10, 11), (11, 12),   # 中指
    (0, 13), (13, 14), (14, 15), (15, 16), # 无名指
    (0, 17), (17, 18), (18, 19), (19, 20), # 小指
    (5, 9), (9, 13), (13, 17),             # 掌心横向连接
]


def draw_landmarks(frame: np.ndarray, landmarks, color=(0, 255, 0)):
    """在 OpenCV 帧上绘制手部骨架（仅用于命令行测试）。"""
    if landmarks is None:
        return
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (255, 0, 0), -1)


# ── 命令行独立测试 ──────────────────────────────────────────────
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("无法打开摄像头")

    print("按 Q 退出")
    with Recognizer() as rec:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            letter, conf = rec.predict(frame)
            draw_landmarks(frame, rec.get_landmarks())

            label = f"{letter}  {conf:.0%}" if letter else (
                f"detecting...  {conf:.0%}" if conf > 0 else "no hand"
            )
            cv2.putText(frame, label, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.imshow("ASL Recognizer - press Q to quit", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

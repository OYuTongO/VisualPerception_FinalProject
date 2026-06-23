import os
import pickle
import collections
import cv2
import numpy as np
import mediapipe as mp
from pythonosc import udp_client
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── MJPEG：把画面通过 HTTP 视频流推给 UE ──
import threading
import time
from flask import Flask, Response

# ── Target Config ────────────────────────────────────────────────
# 按游戏流程顺序排：用户依次要拼的单词
TARGET_WORDS  = ["HI", "GATE"]
GREETING_WORD = "HI"      # 拼对它发 /asl/greet（让 Diana 继续）；其它词发 /asl/trigger（换场景）
word_index    = 0         # 当前在第几个单词
current_index = 0         # 当前单词里第几个字母

# ── GATE 之后的 YES/NO 二选一（选择模式）──
CHOICE_WORDS   = ["YES", "NO"]   # 备选；YES/NO 首字母不同(Y/N)，靠首字母区分走哪条
current_choice = None            # 当前正在拼哪个（None = 还没定）
choice_index   = 0
answered       = False           # 已回答过就不再触发

# ── NO 之后：再让用户比 OKAY，比完切到场景3（结尾阶段）──
ENDING_WORD     = "OKAY"   # NO 之后要拼的词（4 个字母，沿用 4 格 HUD）
ending_index    = 0        # OKAY 里第几个字母
ending_armed    = False    # 是否已进入 OKAY 阶段（NO 回答后开启）
ending_done     = False    # OKAY 拼完就不再触发
ending_arm_time = 0.0      # 何时开始接受 OKAY（给 Diana 说完 joking+anyway 两句留时间）
ENDING_DELAY    = 7.0      # NO 之后等几秒再开始接 OKAY（按 UE 里两句台词的 Delay 调）

# ── OKAY 切场景3之后：最后再让用户比 BYE，比完 Diana 说告别词 → 黑屏 The End ──
BYE_WORD     = "BYE"   # 结尾告别词（3 个字母，沿用 4 格 HUD）
bye_index    = 0       # BYE 里第几个字母
bye_armed    = False   # 是否已进入 BYE 阶段（OKAY 完成后开启）
bye_done     = False   # BYE 拼完就不再触发
bye_arm_time = 0.0     # 何时开始接受 BYE（给 Diana 说场景3台词 + SayBye 两句留时间）
BYE_DELAY    = 15.0    # ready_done 之后等几秒再开始接 BYE（= 5 + UE里SayBye后面那个Delay；UE那个=10 → 这里=15）

OSC_HOST = "127.0.0.1"
OSC_PORT = 8000

STREAM_HOST = "127.0.0.1"
STREAM_PORT = 5000               # UE 端 Media Player 播放 http://127.0.0.1:5000/video_feed

# 最新一帧（JPEG），供视频流读取
_latest_jpeg = None
_jpeg_lock = threading.Lock()

MODEL_PATH      = os.path.join("model", "asl_classifier.pkl")
LANDMARKER_PATH = os.path.join("model", "hand_landmarker.task")
CONFIDENCE_THRESHOLD = 0.6
SMOOTH_FRAMES        = 3


def letter_id(ch: str) -> int:
    ch = ch.upper()
    if ch == " ":
        return 100
    if "A" <= ch <= "Z":
        return ord(ch) - ord("A") + 1
    return 0


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
        self._raw_landmarks = None
        self._smooth_buf = collections.deque(maxlen=SMOOTH_FRAMES)
        self._last_confirmed = None

    def predict(self, frame: np.ndarray):
        self._raw_landmarks = None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_img)

        if not result.hand_landmarks:
            self._smooth_buf.clear()
            self._last_confirmed = None
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

        if len(self._smooth_buf) < SMOOTH_FRAMES or len(set(self._smooth_buf)) != 1:
            self._last_confirmed = None
            return None, confidence

        if letter != self._last_confirmed:
            self._last_confirmed = letter
            return letter, confidence

        return None, confidence

    def get_landmarks(self):
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


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def draw_landmarks(frame: np.ndarray, landmarks, color=(0, 255, 0)):
    if landmarks is None:
        return
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (255, 0, 0), -1)


# ── MJPEG 视频流服务器 ────────────────────────────────────────────
_flask_app = Flask(__name__)


@_flask_app.route("/video_feed")
def _video_feed():
    def gen():
        while True:
            with _jpeg_lock:
                buf = _latest_jpeg
            if buf is not None:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + buf + b"\r\n")
            time.sleep(0.03)

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


def _start_stream_server():
    _flask_app.run(host=STREAM_HOST, port=STREAM_PORT,
                   threaded=True, debug=False, use_reloader=False)


# ── Main ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    client = udp_client.SimpleUDPClient(OSC_HOST, OSC_PORT)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("无法打开摄像头")

    print("感知通讯链路已就绪", flush=True)
    print(f"OSC → {OSC_HOST}:{OSC_PORT}", flush=True)
    print(f"单词序列: {TARGET_WORDS}", flush=True)
    print("按 Q 退出", flush=True)
    print("=" * 50, flush=True)

    # ── 启动 MJPEG 视频流服务器（后台线程）──
    threading.Thread(target=_start_stream_server, daemon=True).start()
    print(f"视频流已启动 → http://{STREAM_HOST}:{STREAM_PORT}/video_feed", flush=True)

    with Recognizer() as rec:
        confirmed_letter = None

        # ── 把识别窗做成"置顶小窗"，叠在 UE 游戏画面的舱室窗口位置 ──
        WIN_NAME = "Hand_Inference"
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_NAME, 480, 360)   # 小窗大小（约 1/12 格位）
        # X 关于屏幕中线镜像（上下不变）：原来左边距40 → 现在右边距40
        import ctypes
        _screen_w = ctypes.windll.user32.GetSystemMetrics(0)
        cv2.moveWindow(WIN_NAME, _screen_w - 40 - 480, 560)  # 右下角（左下角的水平镜像）
        try:
            cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_TOPMOST, 1)  # 始终置顶
        except Exception:
            pass

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            letter, conf = rec.predict(frame)
            draw_landmarks(frame, rec.get_landmarks())

            raw = rec._smooth_buf[-1] if rec._smooth_buf else None

            if letter:
                confirmed_letter = letter

            if letter and word_index < len(TARGET_WORDS):
                target_word = TARGET_WORDS[word_index]
                target_ch = target_word[current_index]

                # ── 分支 A：匹配成功 ─────────────────────────────
                if letter.upper() == target_ch:
                    client.send_message("/asl/progress", current_index)
                    print(f"[OSC] /asl/progress {current_index}  ({target_word}, recognized={letter.upper()}, conf={conf:.0%})", flush=True)
                    current_index += 1

                    if current_index >= len(target_word):
                        # ── 整个单词拼对了 ──
                        if target_word == GREETING_WORD:
                            client.send_message("/asl/greet", 1)
                            print(f"[OSC] /asl/greet 1  ({target_word} 完成 → 让 Diana 继续)", flush=True)
                        else:
                            client.send_message("/asl/trigger", 1)
                            print(f"[OSC] /asl/trigger 1  ({target_word} 完成 → 换场景)", flush=True)
                        # 进入下一个单词
                        word_index += 1
                        current_index = 0
                        if word_index < len(TARGET_WORDS):
                            next_word = TARGET_WORDS[word_index]
                            client.send_message("/asl/set_word", next_word)
                            print(f"[OSC] /asl/set_word {next_word}  (切到下一个词)", flush=True)
                        else:
                            print("所有单词已完成~", flush=True)

                # ── 分支 B：匹配失败 ─────────────────────────────
                else:
                    tid = letter_id(target_ch)
                    client.send_message("/asl/error", tid)
                    print(f"[OSC] /asl/error {tid}  (target={target_ch}, wrong={letter.upper()}, conf={conf:.0%})", flush=True)

            # ── 选择模式：GATE 之后，等用户比 YES 或 NO（逐字母变金，和 GATE 一样）──
            elif letter and word_index >= len(TARGET_WORDS) and not answered:
                L = letter.upper()
                if current_choice is None:
                    # 首字母决定走哪条（Y→YES, N→NO；首字母不同不会混），choice_index 从 0 开始
                    if L == "Y":
                        current_choice, choice_index = "YES", 0
                    elif L == "N":
                        current_choice, choice_index = "NO", 0

                if current_choice is not None:
                    if choice_index < len(current_choice) and L == current_choice[choice_index]:
                        # 这一位拼对 → 让对应那一行的第 choice_index 个字母变金
                        addr = "/asl/yes_progress" if current_choice == "YES" else "/asl/no_progress"
                        client.send_message(addr, choice_index)
                        print(f"[OSC] {addr} {choice_index}  ({current_choice})", flush=True)
                        choice_index += 1
                        if choice_index >= len(current_choice):
                            client.send_message("/asl/answer", current_choice)
                            print(f"[OSC] /asl/answer {current_choice}  (玩家选择)", flush=True)
                            answered = True
                            if current_choice == "NO":
                                # NO → 进入结尾 OKAY 阶段（等 Diana 说完两句话再开始接）
                                ending_armed = True
                                ending_arm_time = time.time() + ENDING_DELAY
                    else:
                        print("[choice] 拼错 → 重置重选", flush=True)
                        current_choice, choice_index = None, 0

            # ── 结尾阶段：NO 之后等用户比 OKAY，逐字母变金，比完发 /asl/ready_done 切到场景3 ──
            elif letter and ending_armed and not ending_done and time.time() >= ending_arm_time:
                L = letter.upper()
                if L == ENDING_WORD[ending_index]:
                    client.send_message("/asl/ready_progress", ending_index)
                    print(f"[OSC] /asl/ready_progress {ending_index}  ({ENDING_WORD})", flush=True)
                    ending_index += 1
                    if ending_index >= len(ENDING_WORD):
                        client.send_message("/asl/ready_done", 1)
                        print("[OSC] /asl/ready_done 1  (OKAY 完成 → 切场景3)", flush=True)
                        ending_done = True
                        # OKAY 完成 → 进入最后的 BYE 阶段（等 Diana 说完场景3台词+SayBye 再开始接）
                        bye_armed = True
                        bye_arm_time = time.time() + BYE_DELAY
                else:
                    print(f"[ending] OKAY 拼错: target={ENDING_WORD[ending_index]}, wrong={L}", flush=True)

            # ── 最终 BYE 阶段：切场景3后等用户比 BYE，比完发 /asl/bye_done → Diana 告别 + 黑屏 The End ──
            elif letter and bye_armed and not bye_done and time.time() >= bye_arm_time:
                L = letter.upper()
                if L == BYE_WORD[bye_index]:
                    client.send_message("/asl/bye_progress", bye_index)
                    print(f"[OSC] /asl/bye_progress {bye_index}  ({BYE_WORD})", flush=True)
                    bye_index += 1
                    if bye_index >= len(BYE_WORD):
                        client.send_message("/asl/bye_done", 1)
                        print("[OSC] /asl/bye_done 1  (BYE 完成 → Diana 告别 + The End)", flush=True)
                        bye_done = True
                else:
                    print(f"[bye] BYE 拼错: target={BYE_WORD[bye_index]}, wrong={L}", flush=True)

            # ── cv2 overlay ───────────────────────────────────────
            if raw:
                cv2.putText(frame, f"now: {raw}  {conf:.0%}", (20, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (200, 200, 200), 3)
            elif conf == 0.0:
                cv2.putText(frame, "no hand", (20, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (100, 100, 100), 3)

            if confirmed_letter:
                cv2.putText(frame, f"confirmed: {confirmed_letter}", (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 4)

            # ── 推流给 UE（MJPEG）：把当前帧编码成 JPEG 供视频流读取 ──
            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                with _jpeg_lock:
                    _latest_jpeg = buf.tobytes()

            cv2.imshow("Hand_Inference", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import pygame

from src.game.learn_mode import ChatLearnMode
from src.recognizer import Recognizer


WINDOW_SIZE = (1280, 720)
FPS = 30


def main():
    project_root = Path(__file__).resolve().parents[2]
    pygame.init()
    pygame.display.set_caption("Diana ASL Chat")
    screen = pygame.display.set_mode(WINDOW_SIZE)
    clock = pygame.time.Clock()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        pygame.quit()
        raise RuntimeError("Could not open webcam. Please check camera permission or device index.")

    recognizer = None
    try:
        recognizer = Recognizer()
        mode = ChatLearnMode(project_root)
        running = True
        while running:
            dt = clock.tick(FPS) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                else:
                    mode.handle_event(event)

            ok, frame = cap.read()
            if not ok:
                frame = None
                letter, confidence, landmarks = None, 0.0, None
            else:
                letter, confidence = recognizer.predict(frame)
                landmarks = recognizer.get_landmarks()

            mode.update(dt, frame, letter, confidence, landmarks)
            mode.draw(screen)
            pygame.display.flip()
    finally:
        cap.release()
        if recognizer is not None:
            recognizer.close()
        pygame.quit()


if __name__ == "__main__":
    main()

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
FPS = 60


def main():
    project_root = Path(__file__).resolve().parents[2]
    pygame.init()
    pygame.display.set_caption("ASL Bridge — Diana Chat")
    screen = pygame.display.set_mode(WINDOW_SIZE)
    clock = pygame.time.Clock()

    # static background — scale to fill, centre-crop, never distort
    bg_path = project_root / "game_module" / "background.png"
    if bg_path.exists():
        _raw = pygame.image.load(str(bg_path)).convert()
        rw, rh = _raw.get_size()
        tw, th = WINDOW_SIZE
        scale = max(tw / rw, th / rh)
        sw, sh = int(rw * scale), int(rh * scale)
        _scaled = pygame.transform.smoothscale(_raw, (sw, sh))
        crop_x = (sw - tw) // 2
        crop_y = (sh - th) // 2
        bg = pygame.Surface(WINDOW_SIZE)
        bg.blit(_scaled, (0, 0), (crop_x, crop_y, tw, th))
    else:
        bg = pygame.Surface(WINDOW_SIZE)
        bg.fill((245, 247, 252))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        pygame.quit()
        raise RuntimeError("Could not open webcam.")

    recognizer = None
    try:
        recognizer = Recognizer()
        mode = ChatLearnMode(project_root)
        running = True
        while running:
            dt = min(clock.tick(FPS) / 1000.0, 0.05)

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

            screen.blit(bg, (0, 0))
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

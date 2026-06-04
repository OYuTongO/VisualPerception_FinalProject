import math
from pathlib import Path

import cv2
import pygame

from src.recognizer import HAND_CONNECTIONS


# ── palette ──────────────────────────────────────────────────────────────────
WHITE        = (255, 255, 255)
LIGHT_GRAY   = (240, 242, 245)
MID_GRAY     = (180, 185, 195)
DARK_TEXT    = (30,  35,  45)
SUBTEXT      = (100, 108, 120)
DIANA_BUBBLE = (220, 236, 255)      # light blue
DIANA_BORDER = (160, 195, 245)
USER_BUBBLE  = (255, 255, 255)      # white
USER_BORDER  = (210, 215, 225)
GOLD         = (255, 196, 48)
GOLD_FILL    = (255, 218, 110)
SLOT_PENDING = (210, 213, 220)
SLOT_ACTIVE  = (90,  160, 255)
ERROR_RED    = (230, 60,  60)
CYAN         = (60,  160, 255)


def draw_round_rect(surface, rect, color, radius=12, border_color=None, border_width=0):
    pygame.draw.rect(surface, color, rect, border_radius=radius)
    if border_color and border_width:
        pygame.draw.rect(surface, border_color, rect, border_width, border_radius=radius)


def load_avatar(path, size):
    avatar_path = Path(path)
    if not avatar_path.exists():
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        pygame.draw.circle(surf, (180, 190, 200), (size // 2, size // 2), size // 2)
        return surf
    image = pygame.image.load(str(avatar_path)).convert_alpha()
    # scale so the shorter side fills `size`, then centre-crop to square
    iw, ih = image.get_size()
    scale = size / min(iw, ih)
    sw, sh = max(size, int(iw * scale)), max(size, int(ih * scale))
    image = pygame.transform.smoothscale(image, (sw, sh))
    cx, cy = (sw - size) // 2, (sh - size) // 2
    # crop to top-third of image for portrait photos (captures face better)
    if sh > sw:
        cy = int(sh * 0.08)
    cropped = pygame.Surface((size, size), pygame.SRCALPHA)
    cropped.blit(image, (0, 0), (cx, cy, size, size))
    mask = pygame.Surface((size, size), pygame.SRCALPHA)
    pygame.draw.circle(mask, (255, 255, 255, 255), (size // 2, size // 2), size // 2)
    result = pygame.Surface((size, size), pygame.SRCALPHA)
    result.blit(cropped, (0, 0))
    result.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    return result


def load_gesture_image(letter, gestures_dir, size):
    """Load ASL gesture jpg from gestures_dir/{letter}_test.jpg, scaled to size."""
    path = Path(gestures_dir) / f"{letter.upper()}_test.jpg"
    if not path.exists():
        return None
    img = pygame.image.load(str(path)).convert()
    return pygame.transform.smoothscale(img, (size, size))


def frame_to_surface(frame, size):
    if frame is None:
        surf = pygame.Surface(size)
        surf.fill((20, 20, 28))
        return surf
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.flip(rgb, 1)
    rgb = cv2.resize(rgb, size)
    return pygame.image.frombuffer(rgb.tobytes(), size, "RGB")


def draw_hand_landmarks(surface, landmarks, size):
    if landmarks is None:
        return
    w, h = size
    pts = [(int((1 - lm.x) * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        pygame.draw.line(surface, CYAN, pts[a], pts[b], 2)
    for p in pts:
        pygame.draw.circle(surface, WHITE, p, 4)
        pygame.draw.circle(surface, CYAN,  p, 2)


def wrap_text(text, font, max_width):
    words = text.split(" ")
    lines, current = [], ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if font.size(candidate)[0] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def draw_wrapped_text(surface, text, font, color, rect, line_spacing=5):
    y = rect.y
    for line in wrap_text(text, font, rect.width):
        rendered = font.render(line, True, color)
        surface.blit(rendered, (rect.x, y))
        y += rendered.get_height() + line_spacing


def draw_letter_slots(surface, text, completed, current_index, font, rect, t):
    """Draw per-letter slots. Returns dict {char_index: center_point}."""
    centers = {}
    slot_w, slot_h, gap = 46, 54, 8
    x, y = rect.x, rect.y

    for i, ch in enumerate(text):
        if ch == " ":
            x += 22
            continue
        if not ch.isalpha():
            rendered = font.render(ch, True, SUBTEXT)
            surface.blit(rendered, (x, y + 12))
            x += rendered.get_width() + 4
            continue
        if x + slot_w > rect.right:
            x = rect.x
            y += slot_h + 14

        slot = pygame.Rect(x, y, slot_w, slot_h)
        filled  = i in completed
        active  = i == current_index

        if filled:
            draw_round_rect(surface, slot, GOLD_FILL, 10, GOLD, 2)
            ch_color = (140, 90, 0)
        elif active:
            pulse = int((math.sin(t * 6) + 1) * 20)
            fill_c = (200 + pulse, 230, 255)
            draw_round_rect(surface, slot, fill_c, 10, SLOT_ACTIVE, 2)
            ch_color = (20, 80, 200)
        else:
            draw_round_rect(surface, slot, LIGHT_GRAY, 10, SLOT_PENDING, 1)
            ch_color = MID_GRAY

        lbl = font.render(ch.upper(), True, ch_color)
        surface.blit(lbl, lbl.get_rect(center=slot.center))
        centers[i] = slot.center
        x += slot_w + gap

    return centers

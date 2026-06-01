import math
from pathlib import Path

import cv2
import pygame

from src.recognizer import HAND_CONNECTIONS


BG_TOP = (8, 13, 32)
BG_BOTTOM = (4, 25, 45)
CYAN = (68, 220, 255)
CYAN_DARK = (22, 92, 126)
WHITE = (235, 245, 255)


def draw_tech_background(screen, t):
    width, height = screen.get_size()
    for y in range(height):
        ratio = y / max(height - 1, 1)
        color = tuple(int(BG_TOP[i] * (1 - ratio) + BG_BOTTOM[i] * ratio) for i in range(3))
        pygame.draw.line(screen, color, (0, y), (width, y))

    offset = int((t * 36) % 48)
    for x in range(-48, width + 48, 48):
        alpha = 55 if (x // 48) % 2 == 0 else 28
        line = pygame.Surface((1, height), pygame.SRCALPHA)
        line.fill((*CYAN_DARK, alpha))
        screen.blit(line, (x + offset, 0))
    for y in range(-48, height + 48, 48):
        alpha = 42 if (y // 48) % 2 == 0 else 24
        line = pygame.Surface((width, 1), pygame.SRCALPHA)
        line.fill((*CYAN_DARK, alpha))
        screen.blit(line, (0, y + offset // 2))

    scan_y = int((math.sin(t * 1.4) * 0.5 + 0.5) * height)
    scan = pygame.Surface((width, 4), pygame.SRCALPHA)
    scan.fill((80, 235, 255, 85))
    screen.blit(scan, (0, scan_y))

    for i in range(34):
        px = int((i * 157 + t * 28) % width)
        py = int((i * 83 + math.sin(t + i) * 18) % height)
        radius = 1 + (i % 3)
        pygame.draw.circle(screen, (60, 210, 255), (px, py), radius)


def draw_round_rect(surface, rect, color, radius=18, border_color=None, border_width=0):
    pygame.draw.rect(surface, color, rect, border_radius=radius)
    if border_color and border_width:
        pygame.draw.rect(surface, border_color, rect, border_width, border_radius=radius)


def load_avatar(path, size):
    avatar_path = Path(path)
    if not avatar_path.exists():
        raise FileNotFoundError(f"Missing avatar file: {avatar_path}. Please place it under ./pnz/.")
    image = pygame.image.load(str(avatar_path)).convert_alpha()
    image = pygame.transform.smoothscale(image, (size, size))

    mask = pygame.Surface((size, size), pygame.SRCALPHA)
    pygame.draw.circle(mask, (255, 255, 255, 255), (size // 2, size // 2), size // 2)
    rounded = pygame.Surface((size, size), pygame.SRCALPHA)
    rounded.blit(image, (0, 0))
    rounded.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    return rounded


def frame_to_surface(frame, size):
    if frame is None:
        surface = pygame.Surface(size)
        surface.fill((12, 18, 32))
        return surface
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.flip(rgb, 1)
    rgb = cv2.resize(rgb, size)
    return pygame.image.frombuffer(rgb.tobytes(), size, "RGB")


def draw_hand_landmarks(surface, landmarks, size):
    if landmarks is None:
        return
    width, height = size
    points = [(int((1 - lm.x) * width), int(lm.y * height)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        pygame.draw.line(surface, CYAN, points[a], points[b], 3)
    for point in points:
        pygame.draw.circle(surface, WHITE, point, 5)
        pygame.draw.circle(surface, CYAN, point, 3)


def wrap_text(text, font, max_width):
    words = text.split(" ")
    lines = []
    current = ""
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


def draw_wrapped_text(surface, text, font, color, rect, line_spacing=6):
    y = rect.y
    for line in wrap_text(text, font, rect.width):
        rendered = font.render(line, True, color)
        surface.blit(rendered, (rect.x, y))
        y += rendered.get_height() + line_spacing


def draw_progress_text(surface, text, completed_indices, current_index, font, rect):
    x = rect.x
    y = rect.y
    max_x = rect.right
    line_height = font.get_height() + 8
    for index, char in enumerate(text):
        if char == "\n":
            x = rect.x
            y += line_height
            continue
        color = (170, 176, 190)
        if index in completed_indices:
            color = (255, 206, 76)
        elif not char.isalpha():
            color = (210, 216, 230)
        rendered = font.render(char, True, color)
        if x + rendered.get_width() > max_x and char != " ":
            x = rect.x
            y += line_height
        surface.blit(rendered, (x, y))
        if index == current_index:
            pygame.draw.line(surface, CYAN, (x, y + rendered.get_height() + 2), (x + rendered.get_width(), y + rendered.get_height() + 2), 2)
        x += rendered.get_width()

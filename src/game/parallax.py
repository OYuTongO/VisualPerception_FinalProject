"""
Parallax background engine + underwater HUD for ASL Bridge.

Classes
-------
ParallaxLayer   — one scrolling layer with per-frame Y-bob distortion
ParallaxBackground — orchestrates three layers + post-processing overlays
UnderwaterHUD   — camera scan panel (bottom-left) and holographic ASL area (centre)

Assets expected in game_module/assets/:
    bg_far.jpg   bg_mid.jpg   bg_fore.jpg
Procedural placeholders are drawn when a file is absent.
"""
from __future__ import annotations

import math
from pathlib import Path

import pygame

# ── constants ────────────────────────────────────────────────────────────────
SW, SH = 1280, 720
CENTER_X = SW // 2

CYAN_GLOW  = (0, 255, 255)
DEEP_SEA   = (10, 26, 47)          # opaque version for fills
DEEP_SEA_A = (10, 26, 47, 180)     # semi-transparent for panels

_TILE_W = SW + 400   # single-tile width (enough to hide one full shift)


# ── procedural placeholder layers ────────────────────────────────────────────
def _procedural(index: int) -> pygame.Surface:
    surf = pygame.Surface((_TILE_W, SH))
    palettes = [
        ((2, 8, 28),  (4, 18, 52)),
        ((4, 14, 38), (8, 32, 68)),
        ((2, 10, 24), (6, 22, 44)),
    ]
    top, bot = palettes[index]
    for y in range(SH):
        r = y / (SH - 1)
        c = tuple(int(top[i] + (bot[i] - top[i]) * r) for i in range(3))
        pygame.draw.line(surf, c, (0, y), (_TILE_W, y))

    if index == 0:
        for i in range(6):
            x = int(_TILE_W * (0.08 + i * 0.17))
            ray = pygame.Surface((56, SH), pygame.SRCALPHA)
            for rx in range(56):
                a = int(20 * math.exp(-((rx - 28) ** 2) / 180))
                pygame.draw.line(ray, (120, 200, 255, a), (rx, 0), (rx, SH))
            surf.blit(ray, (x - 28, 0))
        for x in range(_TILE_W):
            h = int(80 + 40 * math.sin(x * 0.012) + 20 * math.sin(x * 0.031 + 1.2))
            pygame.draw.line(surf, (2, 12, 32), (x, SH - h), (x, SH))

    elif index == 1:
        cx, cy = _TILE_W // 2, SH - 180
        pygame.draw.ellipse(surf, (18, 52, 88), (cx - 220, cy - 130, 440, 260))
        pygame.draw.ellipse(surf, (28, 78, 118), (cx - 220, cy - 130, 440, 260), 3)
        for wx in range(-3, 4):
            wc = (cx + wx * 58, cy - 20)
            pygame.draw.circle(surf, (255, 230, 140), wc, 14)
            pygame.draw.circle(surf, (255, 200, 80), wc, 10)
        for sx in (-160, -80, 0, 80, 160):
            pygame.draw.line(surf, (22, 64, 102), (cx + sx, cy + 90), (cx + sx, SH), 6)
        for x in range(_TILE_W):
            h = int(60 + 30 * math.sin(x * 0.018 + 0.5) + 15 * math.sin(x * 0.044))
            pygame.draw.line(surf, (8, 26, 44), (x, SH - h), (x, SH))

    else:
        for i in range(8):
            rx = int(_TILE_W * (i / 7))
            rh, rw = 60 + (i * 37 % 80), 80 + (i * 53 % 100)
            pygame.draw.ellipse(surf, (6, 20, 34), (rx - rw // 2, SH - rh, rw, rh * 2))

    return surf


def _load_tile(path: Path, index: int) -> pygame.Surface:
    if path.exists():
        img = pygame.image.load(str(path)).convert()
        aspect = img.get_width() / img.get_height()
        tw = max(_TILE_W, int(SH * aspect))
        img = pygame.transform.smoothscale(img, (tw, SH))
        if img.get_width() > _TILE_W:
            cx = (img.get_width() - _TILE_W) // 2
            out = pygame.Surface((_TILE_W, SH))
            out.blit(img, (0, 0), (cx, 0, _TILE_W, SH))
            return out
        return img
    return _procedural(index)


# ── ParallaxLayer ─────────────────────────────────────────────────────────────
class ParallaxLayer:
    """One scrolling background layer with Y-bob wave distortion."""

    # each layer bobs at a slightly different phase and speed so they
    # never look synchronised
    _BOB_PARAMS = (
        (0.4,  0.9,  3.0),   # far:  amplitude px, freq, phase
        (0.7,  0.7,  1.4),   # mid
        (1.1,  0.55, 0.0),   # fore
    )

    def __init__(self, path: Path, index: int, speed: float):
        self.speed = speed
        self._index = index
        tile = _load_tile(path, index)
        self._tile_w = tile.get_width()
        # double-tile for seamless wrapping
        self._strip = pygame.Surface((self._tile_w * 2, SH))
        self._strip.blit(tile, (0, 0))
        self._strip.blit(tile, (self._tile_w, 0))

        self._scroll = 0.0   # smooth scrolled offset
        self._target = 0.0

    def set_target(self, mouse_x: int, max_shift: int = 180):
        norm = (mouse_x / SW) * 2.0 - 1.0
        self._target = norm * max_shift * self.speed

    def update(self, dt: float):
        alpha = 1.0 - math.exp(-dt / 0.18)
        self._scroll += (self._target - self._scroll) * alpha

    def draw(self, screen: pygame.Surface, t: float):
        amp, freq, phase = self._BOB_PARAMS[self._index]
        bob = int(amp * math.sin(freq * t + phase))
        blit_x = (int(self._scroll) % self._tile_w) - self._tile_w
        screen.blit(self._strip, (blit_x, bob))


# ── ParallaxBackground ────────────────────────────────────────────────────────
class ParallaxBackground:
    _SPEEDS = (0.1, 0.3, 0.6)
    _NAMES  = ("bg_far.jpg", "bg_mid.jpg", "bg_fore.jpg")

    def __init__(self, assets_dir: Path):
        self._layers = [
            ParallaxLayer(assets_dir / name, i, spd)
            for i, (name, spd) in enumerate(zip(self._NAMES, self._SPEEDS))
        ]
        self._t = 0.0
        self._ripple = pygame.Surface((SW, SH), pygame.SRCALPHA)
        self._vignette = self._build_vignette()

    # pre-bake a static vertical gradient overlay (deep-blue → transparent)
    @staticmethod
    def _build_vignette() -> pygame.Surface:
        surf = pygame.Surface((SW, SH), pygame.SRCALPHA)
        for y in range(SH):
            ratio = y / (SH - 1)
            # strong at top (dark deep blue), fades to zero at 60 % down
            alpha = int(max(0, 72 * (1 - ratio / 0.6)))
            pygame.draw.line(surf, (8, 18, 48, alpha), (0, y), (SW, y))
        return surf

    def update(self, dt: float, mouse_x: int):
        self._t += dt
        for layer in self._layers:
            layer.set_target(mouse_x)
            layer.update(dt)

    def draw(self, screen: pygame.Surface):
        for layer in self._layers:
            layer.draw(screen, self._t)
        self._draw_ripple(screen)
        self._draw_caustics(screen)
        screen.blit(self._vignette, (0, 0))   # vertical deep-blue gradient on top

    def _draw_ripple(self, screen: pygame.Surface):
        t = self._t
        ov = self._ripple
        ov.fill((0, 0, 0, 0))
        for band in range(10):
            phase = band * (math.pi * 2 / 10)
            y = int((SH * 0.5) + math.sin(t * 0.52 + phase) * SH * 0.44)
            a = 9 + int(5 * math.sin(t * 1.05 + phase))
            pygame.draw.line(ov, (80, 160, 255, a), (0, y), (SW, y), 16)
        pa = int(7 + 5 * math.sin(t * 0.38))
        ov.fill((20, 70, 160, pa), special_flags=pygame.BLEND_RGBA_ADD)
        screen.blit(ov, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)

    def _draw_caustics(self, screen: pygame.Surface):
        t = self._t
        for i in range(16):
            cx = int(SW * 0.5 + math.sin(t * 0.36 + i * 1.73) * SW * 0.46)
            cy = int(SH * 0.36 + math.cos(t * 0.27 + i * 2.11) * SH * 0.26)
            r  = 26 + int(11 * math.sin(t * 1.1 + i))
            sp = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            pygame.draw.circle(sp, (140, 210, 255, 13), (r, r), r)
            pygame.draw.circle(sp, (190, 230, 255,  7), (r, r), r // 2)
            screen.blit(sp, (cx - r, cy - r), special_flags=pygame.BLEND_RGBA_ADD)


# ── UnderwaterHUD ─────────────────────────────────────────────────────────────
class UnderwaterHUD:
    """
    Draws two placeholder UI zones on top of the parallax background:
      - Camera panel  : bottom-left, scanline texture, L-corner brackets
      - Hologram zone : centre screen, animated horizontal scan band
    """
    CAM_RECT  = pygame.Rect(32, SH - 272, 380, 240)
    HOLO_RECT = pygame.Rect(SW // 2 - 200, SH // 2 - 160, 400, 320)

    def __init__(self):
        self._t = 0.0
        self._scanlines = self._build_scanlines(self.CAM_RECT.size)

    @staticmethod
    def _build_scanlines(size: tuple[int, int]) -> pygame.Surface:
        w, h = size
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        for y in range(0, h, 4):
            pygame.draw.line(surf, (0, 0, 0, 55), (0, y), (w, y), 1)
        return surf

    def update(self, dt: float):
        self._t += dt

    def draw(self, screen: pygame.Surface):
        self._draw_camera_panel(screen)
        self._draw_holo_zone(screen)

    # ── camera panel ──────────────────────────────────────────────────────────
    def _draw_camera_panel(self, screen: pygame.Surface):
        r = self.CAM_RECT
        # semi-transparent fill
        panel = pygame.Surface(r.size, pygame.SRCALPHA)
        panel.fill((*DEEP_SEA, 180))
        screen.blit(panel, r.topleft)
        # scanlines over fill
        screen.blit(self._scanlines, r.topleft)
        # CYAN_GLOW border  (1 px)
        pygame.draw.rect(screen, CYAN_GLOW, r, 1, border_radius=4)
        # L-corner brackets (8×8 px each corner)
        self._draw_brackets(screen, r, CYAN_GLOW, arm=14, thickness=2)
        # label
        font = pygame.font.SysFont("arial", 16, bold=True)
        label = font.render("CAPTURE FEED", True, CYAN_GLOW)
        screen.blit(label, (r.x + 10, r.y + 8))

    # ── hologram zone ─────────────────────────────────────────────────────────
    def _draw_holo_zone(self, screen: pygame.Surface):
        r = self.HOLO_RECT
        # translucent fill
        panel = pygame.Surface(r.size, pygame.SRCALPHA)
        panel.fill((*DEEP_SEA, 140))
        screen.blit(panel, r.topleft)
        # border
        pygame.draw.rect(screen, CYAN_GLOW, r, 1, border_radius=6)
        self._draw_brackets(screen, r, CYAN_GLOW, arm=18, thickness=2)

        # animated horizontal scan band (Avengers-style HUD sweep)
        scan_y = r.y + int((r.height * 0.5) + math.sin(self._t * 1.1) * r.height * 0.42)
        band = pygame.Surface((r.width, 6), pygame.SRCALPHA)
        for bx in range(r.width):
            ratio = abs(bx - r.width // 2) / (r.width // 2)
            a = int((1 - ratio) * 90)
            band.set_at((bx, 3), (*CYAN_GLOW, a))
            band.set_at((bx, 2), (*CYAN_GLOW, a // 3))
            band.set_at((bx, 4), (*CYAN_GLOW, a // 3))
        screen.blit(band, (r.x, scan_y - 3))

        # label
        font = pygame.font.SysFont("arial", 16, bold=True)
        label = font.render("ASL HOLOGRAM", True, CYAN_GLOW)
        screen.blit(label, (r.x + 10, r.y + 10))

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _draw_brackets(
        surface: pygame.Surface,
        rect: pygame.Rect,
        color: tuple,
        arm: int = 12,
        thickness: int = 2,
    ):
        corners = [
            (rect.left,       rect.top,     +1, +1),
            (rect.right - 1,  rect.top,     -1, +1),
            (rect.left,       rect.bottom-1, +1, -1),
            (rect.right - 1,  rect.bottom-1, -1, -1),
        ]
        for x, y, sx, sy in corners:
            pygame.draw.line(surface, color, (x, y), (x + sx * arm, y), thickness)
            pygame.draw.line(surface, color, (x, y), (x, y + sy * arm), thickness)

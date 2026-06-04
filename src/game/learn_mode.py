import math
from pathlib import Path

import pygame

from src.game.dialogue_script import DIALOGUE_SCRIPT, validate_dialogue
from src.utils.draw import (
    DARK_TEXT, DIANA_BUBBLE, DIANA_BORDER, ERROR_RED, GOLD,
    LIGHT_GRAY, MID_GRAY, SUBTEXT, USER_BUBBLE, USER_BORDER, WHITE,
    draw_hand_landmarks, draw_letter_slots, draw_round_rect,
    draw_wrapped_text, frame_to_surface, load_avatar, load_gesture_image,
)

WIDTH,  HEIGHT  = 1280, 720
TOPBAR_H        = 56          # recognition status bar
CAM_W, CAM_H   = 220, 165     # small camera preview (bottom-right)
CHAT_X          = 20          # chat column left edge
CHAT_W          = WIDTH - 40  # chat column width
AVATAR_SIZE     = 52
BUBBLE_RADIUS   = 18
GESTURE_SIZE    = 160         # gesture sticker size in pixels


# ── colour helpers ────────────────────────────────────────────────────────────
_TOPBAR_BG   = (248, 249, 252, 230)
_CHAT_BG     = (0, 0, 0, 0)          # transparent — background shows through
_ERR_BADGE   = (230, 50, 50)
_ERR_TEXT    = (255, 255, 255)
_FINISH_OVERLAY = (20, 24, 36, 200)
_FINISH_CARD    = (255, 255, 255)
_FINISH_BORDER  = (255, 196, 48)


class ChatLearnMode:
    def __init__(self, project_root: Path):
        validate_dialogue()
        self.project_root    = project_root
        self.gestures_dir    = project_root / "game_module" / "asl_alphabet_test"
        self.dialogue        = DIALOGUE_SCRIPT
        self.diana_avatar    = load_avatar(project_root / "game_module" / "D.jpg", AVATAR_SIZE)
        self.user_avatar     = load_avatar(project_root / "game_module" / "X.jpg", AVATAR_SIZE)

        # fonts
        self.font_ui    = pygame.font.SysFont("arial", 17)
        self.font_chat  = pygame.font.SysFont("arial", 22)
        self.font_slot  = pygame.font.SysFont("arial", 24, bold=True)
        self.font_label = pygame.font.SysFont("arial", 14)
        self.font_big   = pygame.font.SysFont("arial", 72, bold=True)
        self.font_med   = pygame.font.SysFont("arial", 26, bold=True)
        self.font_small = pygame.font.SysFont("arial", 19)

        # gesture image cache
        self._gesture_cache: dict[str, pygame.Surface | None] = {}

        # state
        self.dialogue_index   = 0
        self.phase            = "DIANA_TYPING"   # phases below
        self.typed_count      = 0
        self.type_timer       = 0.0
        self.type_speed       = 28.0             # chars per second
        self.user_char_index  = 0
        self.completed        = set()
        self.total_errors     = 0
        self.time             = 0.0
        self.wrong_shake_until = 0.0
        self.correct_flash_until = 0.0
        self.error_demo_until = 0.0
        self.error_demo_letter: str | None = None
        self.frame            = None
        self.landmarks        = None
        self.last_letter: str | None = None
        self.last_confidence  = 0.0

        # chat message log — each entry: dict with keys type/text/letter/slots
        # type: "diana_text" | "user_slots" | "diana_sticker"
        self._messages: list[dict] = []
        self._slot_centers: dict[int, tuple] = {}

        self._skip_non_letters()

    # ── public API ────────────────────────────────────────────────────────────
    def handle_event(self, event):
        return None

    def update(self, dt, frame, letter, confidence, landmarks):
        self.time       += dt
        self.frame       = frame
        self.landmarks   = landmarks
        self.last_letter = letter
        self.last_confidence = confidence

        if self.phase == "DIANA_TYPING":
            self._update_diana_typing(dt)
        elif self.phase == "USER_TYPING":
            self._update_user_typing(dt)
        elif self.phase == "WAIT_SIGN":
            self._update_signing(letter)
        elif self.phase == "ERROR_DEMO":
            if self.time >= self.error_demo_until:
                self.phase = "WAIT_SIGN"
                self.error_demo_letter = None
        elif self.phase == "ROUND_DONE":
            if self.time >= self.correct_flash_until:
                self._advance_dialogue()

    def draw(self, screen: pygame.Surface):
        self._draw_topbar(screen)
        self._draw_chat(screen)
        self._draw_camera(screen)
        if self.phase == "FINISHED":
            self._draw_finished(screen)

    # ── dialogue state machine ────────────────────────────────────────────────
    def _current_item(self):
        return self.dialogue[min(self.dialogue_index, len(self.dialogue) - 1)]

    def _current_user_text(self):
        return self._current_item()["user"]

    def _current_target(self) -> str | None:
        text = self._current_user_text()
        if self.user_char_index < len(text) and text[self.user_char_index].isalpha():
            return text[self.user_char_index].upper()
        return None

    def _skip_non_letters(self):
        if self.dialogue_index >= len(self.dialogue):
            return
        text = self._current_user_text()
        while self.user_char_index < len(text) and not text[self.user_char_index].isalpha():
            self.completed.add(self.user_char_index)
            self.user_char_index += 1

    def _update_diana_typing(self, dt):
        text = self._current_item()["diana"]
        self.type_timer += dt * self.type_speed
        self.typed_count = min(len(text), int(self.type_timer))
        if self.typed_count >= len(text):
            # push completed Diana message into log
            self._push_diana_text(text)
            self.phase       = "USER_TYPING"
            self.typed_count = 0
            self.type_timer  = 0.0

    def _update_user_typing(self, dt):
        text = self._current_user_text()
        self.type_timer += dt * self.type_speed
        self.typed_count = min(len(text), int(self.type_timer))
        if self.typed_count >= len(text):
            # push user slot bubble into log
            self._push_user_slots(text)
            self.phase = "WAIT_SIGN"
            self._skip_non_letters()

    def _update_signing(self, letter):
        if not letter:
            return
        target = self._current_target()
        if target is None:
            self._complete_reply()
            return
        detected = letter.upper()
        if detected == target:
            self.completed.add(self.user_char_index)
            self.user_char_index += 1
            self._skip_non_letters()
            # update the live user-slots message
            self._update_user_slots_message()
            if self.user_char_index >= len(self._current_user_text()):
                self._complete_reply()
        else:
            self.total_errors += 1
            self.wrong_shake_until = self.time + 0.5
            # Diana sends the sticker for the target letter
            self._push_diana_sticker(target)
            self.error_demo_letter = target
            self.error_demo_until  = self.time + 2.2
            self.phase = "ERROR_DEMO"

    def _complete_reply(self):
        self.completed.update(range(len(self._current_user_text())))
        self._update_user_slots_message()
        self.phase = "ROUND_DONE"
        self.correct_flash_until = self.time + 1.1

    def _advance_dialogue(self):
        self.dialogue_index += 1
        if self.dialogue_index >= len(self.dialogue):
            self.phase = "FINISHED"
            return
        self.phase           = "DIANA_TYPING"
        self.typed_count     = 0
        self.type_timer      = 0.0
        self.user_char_index = 0
        self.completed       = set()
        self.error_demo_letter = None
        self._skip_non_letters()

    # ── message log helpers ───────────────────────────────────────────────────
    def _push_diana_text(self, text: str):
        self._messages.append({"type": "diana_text", "text": text})

    def _push_diana_sticker(self, letter: str):
        self._messages.append({"type": "diana_sticker", "letter": letter})

    def _push_user_slots(self, text: str):
        entry = {"type": "user_slots", "text": text, "completed": set(self.completed)}
        self._messages.append(entry)

    def _update_user_slots_message(self):
        # update the most recent user_slots entry with current completed set
        for msg in reversed(self._messages):
            if msg["type"] == "user_slots":
                msg["completed"] = set(self.completed)
                break

    def _get_gesture(self, letter: str) -> pygame.Surface | None:
        if letter not in self._gesture_cache:
            self._gesture_cache[letter] = load_gesture_image(
                letter, self.gestures_dir, GESTURE_SIZE
            )
        return self._gesture_cache[letter]

    # ── drawing ───────────────────────────────────────────────────────────────
    def _draw_topbar(self, screen: pygame.Surface):
        bar = pygame.Surface((WIDTH, TOPBAR_H), pygame.SRCALPHA)
        bar.fill((255, 255, 255, 220))
        screen.blit(bar, (0, 0))
        pygame.draw.line(screen, (210, 215, 225), (0, TOPBAR_H), (WIDTH, TOPBAR_H), 1)

        # title
        title = self.font_med.render("ASL Bridge — Diana Chat", True, (30, 35, 45))
        screen.blit(title, (20, TOPBAR_H // 2 - title.get_height() // 2))

        # target letter indicator
        target = self._current_target() or "-"
        target_surf = self.font_ui.render(f"Target: {target}", True, (40, 100, 220))
        screen.blit(target_surf, (440, TOPBAR_H // 2 - target_surf.get_height() // 2))

        # detected letter
        det = self.last_letter or "-"
        conf = f"{self.last_confidence:.0%}"
        det_surf = self.font_ui.render(f"Detected: {det}  {conf}", True, SUBTEXT)
        screen.blit(det_surf, (600, TOPBAR_H // 2 - det_surf.get_height() // 2))

        # error badge
        self._draw_error_badge(screen)

    def _draw_error_badge(self, screen: pygame.Surface):
        shake_x = 0
        if self.time < self.wrong_shake_until:
            shake_x = int(math.sin(self.time * 60) * 4)

        badge_font = pygame.font.SysFont("arial", 20, bold=True)
        label = badge_font.render(f"✕ {self.total_errors}", True, _ERR_TEXT)
        pad_x, pad_y = 18, 6
        bw = label.get_width() + pad_x * 2
        bh = label.get_height() + pad_y * 2
        bx = WIDTH - bw - 24 + shake_x
        by = (TOPBAR_H - bh) // 2

        badge = pygame.Surface((bw, bh), pygame.SRCALPHA)
        pygame.draw.rect(badge, (*_ERR_BADGE, 230), badge.get_rect(), border_radius=bh // 2)
        badge.blit(label, (pad_x, pad_y))
        screen.blit(badge, (bx, by))

        caption = self.font_label.render("ERRORS", True, SUBTEXT)
        screen.blit(caption, (bx + bw // 2 - caption.get_width() // 2, by + bh + 2))

    def _draw_chat(self, screen: pygame.Surface):
        chat_top = TOPBAR_H + 12
        chat_bottom = HEIGHT - 12
        y = chat_top

        # If Diana is currently typing, show live preview bubble
        all_messages = list(self._messages)
        if self.phase == "DIANA_TYPING":
            live_text = self._current_item()["diana"][:self.typed_count]
            all_messages.append({"type": "diana_text", "text": live_text, "_live": True})

        for msg in all_messages:
            if y >= chat_bottom:
                break
            mtype = msg["type"]
            if mtype == "diana_text":
                h = self._bubble_height_text(msg["text"])
                if y + h <= chat_bottom:
                    self._draw_diana_text_bubble(screen, msg["text"], y)
                    y += h + 12
            elif mtype == "diana_sticker":
                h = GESTURE_SIZE + 32
                if y + h <= chat_bottom:
                    self._draw_diana_sticker_bubble(screen, msg["letter"], y)
                    y += h + 12
            elif mtype == "user_slots":
                h = self._bubble_height_slots(msg["text"])
                if y + h <= chat_bottom:
                    cur = self.user_char_index if self.phase in {"WAIT_SIGN", "ERROR_DEMO"} else -1
                    self._draw_user_slots_bubble(screen, msg["text"], msg["completed"], cur, y)
                    y += h + 12

    def _bubble_height_text(self, text: str) -> int:
        lines = 1
        test_surf = self.font_chat.render(text, True, DARK_TEXT)
        max_w = CHAT_W - AVATAR_SIZE - 48 - 40
        if test_surf.get_width() > max_w:
            lines = math.ceil(test_surf.get_width() / max_w) + 1
        return AVATAR_SIZE + max(0, (lines - 1) * (self.font_chat.get_height() + 5)) + 12

    def _bubble_height_slots(self, text: str) -> int:
        letters = sum(1 for c in text if c.isalpha())
        slots_w = CHAT_W - AVATAR_SIZE - 48 - 40
        slots_per_row = max(1, slots_w // (46 + 8))
        rows = math.ceil(letters / slots_per_row) if letters else 1
        return rows * (54 + 14) + 32

    def _draw_diana_text_bubble(self, screen: pygame.Surface, text: str, y: int):
        ax = CHAT_X
        ay = y
        screen.blit(self.diana_avatar, (ax, ay))
        pygame.draw.circle(screen, DIANA_BORDER, (ax + AVATAR_SIZE // 2, ay + AVATAR_SIZE // 2), AVATAR_SIZE // 2, 2)

        bx = ax + AVATAR_SIZE + 10
        max_w = CHAT_W - AVATAR_SIZE - 20
        lines = wrap_lines(text, self.font_chat, max_w - 32)
        bh = len(lines) * (self.font_chat.get_height() + 5) + 24
        bw = min(max_w, max((self.font_chat.size(l)[0] for l in lines), default=60) + 32)
        bubble = pygame.Rect(bx, ay, bw, max(bh, AVATAR_SIZE))
        b_surf = pygame.Surface(bubble.size, pygame.SRCALPHA)
        draw_round_rect(b_surf, b_surf.get_rect(), (*DIANA_BUBBLE, 230), BUBBLE_RADIUS, DIANA_BORDER, 1)
        screen.blit(b_surf, bubble.topleft)

        # tail
        tail = [(bx, ay + 14), (bx - 10, ay + 20), (bx, ay + 28)]
        pygame.draw.polygon(screen, DIANA_BUBBLE, tail)

        ty = ay + 12
        for line in lines:
            rendered = self.font_chat.render(line, True, DARK_TEXT)
            screen.blit(rendered, (bx + 16, ty))
            ty += self.font_chat.get_height() + 5

    def _draw_diana_sticker_bubble(self, screen: pygame.Surface, letter: str, y: int):
        ax = CHAT_X
        ay = y
        screen.blit(self.diana_avatar, (ax, ay))
        pygame.draw.circle(screen, DIANA_BORDER, (ax + AVATAR_SIZE // 2, ay + AVATAR_SIZE // 2), AVATAR_SIZE // 2, 2)

        bx = ax + AVATAR_SIZE + 10
        pad = 12
        bw = GESTURE_SIZE + pad * 2
        bh = GESTURE_SIZE + pad * 2 + self.font_label.get_height() + 8
        bubble = pygame.Rect(bx, ay, bw, bh)
        b_surf = pygame.Surface(bubble.size, pygame.SRCALPHA)
        draw_round_rect(b_surf, b_surf.get_rect(), (*DIANA_BUBBLE, 220), BUBBLE_RADIUS, DIANA_BORDER, 1)
        screen.blit(b_surf, bubble.topleft)

        # tail
        tail = [(bx, ay + 14), (bx - 10, ay + 20), (bx, ay + 28)]
        pygame.draw.polygon(screen, DIANA_BUBBLE, tail)

        gesture_img = self._get_gesture(letter)
        if gesture_img:
            screen.blit(gesture_img, (bx + pad, ay + pad))
        else:
            placeholder = pygame.Rect(bx + pad, ay + pad, GESTURE_SIZE, GESTURE_SIZE)
            draw_round_rect(screen, placeholder, LIGHT_GRAY, 8, MID_GRAY, 1)
            lbl = self.font_med.render(letter, True, (100, 120, 180))
            screen.blit(lbl, lbl.get_rect(center=placeholder.center))

        caption = self.font_label.render(f"Sign  {letter}", True, SUBTEXT)
        screen.blit(caption, (bx + pad, ay + pad + GESTURE_SIZE + 6))

    def _draw_user_slots_bubble(self, screen: pygame.Surface, text: str, completed: set, current_index: int, y: int):
        max_w = CHAT_W - AVATAR_SIZE - 20
        bh = self._bubble_height_slots(text) + 8
        bw = min(max_w, 46 * min(len([c for c in text if c.isalpha()]), 8) + 8 * 7 + 32)
        bw = max(bw, 100)
        bx = WIDTH - CHAT_X - AVATAR_SIZE - 10 - bw
        ay = y
        ax = WIDTH - CHAT_X - AVATAR_SIZE

        bubble = pygame.Rect(bx, ay, bw, bh)
        b_surf = pygame.Surface(bubble.size, pygame.SRCALPHA)
        draw_round_rect(b_surf, b_surf.get_rect(), (*USER_BUBBLE, 235), BUBBLE_RADIUS, USER_BORDER, 1)
        screen.blit(b_surf, bubble.topleft)

        # tail
        tail = [(bx + bw, ay + 14), (bx + bw + 10, ay + 20), (bx + bw, ay + 28)]
        pygame.draw.polygon(screen, USER_BUBBLE, tail)

        slots_rect = pygame.Rect(bx + 16, ay + 14, bw - 32, bh - 20)
        self._slot_centers = draw_letter_slots(screen, text, completed, current_index, self.font_slot, slots_rect, self.time)

        screen.blit(self.user_avatar, (ax, ay))
        pygame.draw.circle(screen, USER_BORDER, (ax + AVATAR_SIZE // 2, ay + AVATAR_SIZE // 2), AVATAR_SIZE // 2, 2)

    def _draw_camera(self, screen: pygame.Surface):
        cam_x = WIDTH - CAM_W - 12
        cam_y = HEIGHT - CAM_H - 12
        cam_surf = frame_to_surface(self.frame, (CAM_W, CAM_H))
        draw_hand_landmarks(cam_surf, self.landmarks, (CAM_W, CAM_H))

        border = pygame.Rect(cam_x - 2, cam_y - 2, CAM_W + 4, CAM_H + 4)
        draw_round_rect(screen, border, (200, 210, 230), 10)
        screen.blit(cam_surf, (cam_x, cam_y))
        pygame.draw.rect(screen, (160, 180, 220), (cam_x, cam_y, CAM_W, CAM_H), 2, border_radius=8)

        cam_label = self.font_label.render("Camera", True, SUBTEXT)
        screen.blit(cam_label, (cam_x + 6, cam_y + 4))

    def _draw_finished(self, screen: pygame.Surface):
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill(_FINISH_OVERLAY)
        screen.blit(overlay, (0, 0))

        cw, ch = 620, 300
        cx, cy = (WIDTH - cw) // 2, (HEIGHT - ch) // 2
        card = pygame.Surface((cw, ch), pygame.SRCALPHA)
        draw_round_rect(card, card.get_rect(), (*_FINISH_CARD, 250), 28, _FINISH_BORDER, 3)
        screen.blit(card, (cx, cy))

        title = self.font_big.render("MISSION CLEAR", True, (200, 140, 0))
        screen.blit(title, title.get_rect(center=(WIDTH // 2, cy + 90)))

        sub = self.font_med.render("You completed the ASL dialogue with Diana!", True, (50, 60, 80))
        screen.blit(sub, sub.get_rect(center=(WIDTH // 2, cy + 170)))

        err = self.font_small.render(f"Total errors: {self.total_errors}", True, ERROR_RED)
        screen.blit(err, err.get_rect(center=(WIDTH // 2, cy + 218)))

        hint = self.font_label.render("Press Q or ESC to exit", True, SUBTEXT)
        screen.blit(hint, hint.get_rect(center=(WIDTH // 2, cy + 262)))


def wrap_lines(text, font, max_width):
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

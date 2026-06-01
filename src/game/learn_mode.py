import pygame

from src.game.dialogue_script import DIALOGUE_SCRIPT, validate_dialogue
from src.utils.draw import (
    draw_hand_landmarks,
    draw_progress_text,
    draw_round_rect,
    draw_tech_background,
    draw_wrapped_text,
    frame_to_surface,
    load_avatar,
)


WIDTH, HEIGHT = 1280, 720
CAMERA_RECT = pygame.Rect(38, 88, 520, 390)
CHAT_RECT = pygame.Rect(600, 44, 640, 620)


class ChatLearnMode:
    def __init__(self, project_root):
        validate_dialogue()
        self.project_root = project_root
        self.dialogue = DIALOGUE_SCRIPT
        self.avatar_size = 72
        self.diana_avatar = load_avatar(project_root / "pnz" / "D.jpg", self.avatar_size)
        self.user_avatar = load_avatar(project_root / "pnz" / "X.jpg", self.avatar_size)

        self.title_font = pygame.font.SysFont("arial", 30, bold=True)
        self.hud_font = pygame.font.SysFont("arial", 24, bold=True)
        self.chat_font = pygame.font.SysFont("arial", 25)
        self.user_font = pygame.font.SysFont("arial", 30, bold=True)
        self.big_font = pygame.font.SysFont("arial", 96, bold=True)
        self.small_font = pygame.font.SysFont("arial", 20)

        self.dialogue_index = 0
        self.phase = "DIANA_TYPING"
        self.typed_count = 0
        self.type_timer = 0.0
        self.type_speed = 32.0
        self.user_char_index = 0
        self.completed_indices = set()
        self.total_errors = 0
        self.wrong_until = 0.0
        self.correct_until = 0.0
        self.wrong_cooldown_until = 0.0
        self.time = 0.0
        self.frame = None
        self.landmarks = None
        self.last_letter = None
        self.last_confidence = 0.0
        self._skip_non_letters()

    def handle_event(self, event):
        return None

    def update(self, dt, frame, letter, confidence, landmarks):
        self.time += dt
        self.frame = frame
        self.landmarks = landmarks
        self.last_letter = letter
        self.last_confidence = confidence

        if self.phase in {"DIANA_TYPING", "USER_TYPING"}:
            self._update_typing(dt)
        elif self.phase == "WAIT_SIGN":
            self._update_signing(letter)
        elif self.phase == "ROUND_DONE" and self.time >= self.correct_until:
            self.dialogue_index += 1
            if self.dialogue_index >= len(self.dialogue):
                self.phase = "FINISHED"
            else:
                self.phase = "DIANA_TYPING"
                self.typed_count = 0
                self.type_timer = 0.0
                self.user_char_index = 0
                self.completed_indices = set()
                self._skip_non_letters()

    def draw(self, screen):
        draw_tech_background(screen, self.time)
        self._draw_title(screen)
        self._draw_camera_panel(screen)
        self._draw_chat_panel(screen)
        self._draw_feedback(screen)
        if self.phase == "FINISHED":
            self._draw_finished(screen)

    def _current_item(self):
        return self.dialogue[min(self.dialogue_index, len(self.dialogue) - 1)]

    def _current_user_text(self):
        return self._current_item()["user"]

    def _current_target(self):
        text = self._current_user_text()
        if self.user_char_index < len(text) and text[self.user_char_index].isalpha():
            return text[self.user_char_index].upper()
        return None

    def _skip_non_letters(self):
        if self.dialogue_index >= len(self.dialogue):
            return
        text = self._current_user_text()
        while self.user_char_index < len(text) and not text[self.user_char_index].isalpha():
            self.completed_indices.add(self.user_char_index)
            self.user_char_index += 1

    def _update_typing(self, dt):
        text = self._current_item()["diana"] if self.phase == "DIANA_TYPING" else self._current_user_text()
        self.type_timer += dt * self.type_speed
        new_count = min(len(text), int(self.type_timer))
        self.typed_count = max(self.typed_count, new_count)
        if self.typed_count >= len(text):
            if self.phase == "DIANA_TYPING":
                self.phase = "USER_TYPING"
                self.typed_count = 0
                self.type_timer = 0.0
            else:
                self.phase = "WAIT_SIGN"
                self._skip_non_letters()

    def _update_signing(self, letter):
        if not letter:
            return
        target = self._current_target()
        if target is None:
            self._complete_current_reply()
            return
        if letter.upper() == target:
            self.completed_indices.add(self.user_char_index)
            self.user_char_index += 1
            self._skip_non_letters()
            if self.user_char_index >= len(self._current_user_text()):
                self._complete_current_reply()
        elif self.time >= self.wrong_cooldown_until:
            self.total_errors += 1
            self.wrong_until = self.time + 0.55
            self.wrong_cooldown_until = self.time + 0.55

    def _complete_current_reply(self):
        self.completed_indices.update(range(len(self._current_user_text())))
        self.phase = "ROUND_DONE"
        self.correct_until = self.time + 1.0

    def _draw_title(self, screen):
        title = self.title_font.render("Diana ASL Chat", True, (230, 248, 255))
        screen.blit(title, (42, 28))
        subtitle = self.small_font.render("Sign each highlighted letter to fill your reply.", True, (120, 210, 235))
        screen.blit(subtitle, (286, 38))

    def _draw_camera_panel(self, screen):
        panel = CAMERA_RECT.inflate(24, 74)
        draw_round_rect(screen, panel, (7, 18, 34), 24, (56, 192, 230), 2)
        camera_surface = frame_to_surface(self.frame, CAMERA_RECT.size)
        draw_hand_landmarks(camera_surface, self.landmarks, CAMERA_RECT.size)
        screen.blit(camera_surface, CAMERA_RECT.topleft)
        pygame.draw.rect(screen, (80, 230, 255), CAMERA_RECT, 2, border_radius=14)

        target = self._current_target() if self.phase != "FINISHED" else None
        target_text = f"Target: {target or '-'}"
        detect_text = f"Detected: {self.last_letter or '-'}  {self.last_confidence:.0%}"
        error_text = f"Errors: {self.total_errors}"
        screen.blit(self.hud_font.render(target_text, True, (255, 206, 76)), (50, 505))
        screen.blit(self.hud_font.render(detect_text, True, (210, 238, 255)), (220, 505))
        screen.blit(self.hud_font.render(error_text, True, (255, 118, 118)), (430, 505))

    def _draw_chat_panel(self, screen):
        draw_round_rect(screen, CHAT_RECT, (12, 22, 42), 26, (40, 160, 202), 2)
        header = self.hud_font.render("Secure chat with Diana", True, (218, 246, 255))
        screen.blit(header, (CHAT_RECT.x + 24, CHAT_RECT.y + 18))
        pygame.draw.line(screen, (38, 118, 152), (CHAT_RECT.x + 20, CHAT_RECT.y + 58), (CHAT_RECT.right - 20, CHAT_RECT.y + 58), 1)

        item = self._current_item()
        diana_text = item["diana"]
        user_text = item["user"]
        if self.phase == "DIANA_TYPING":
            diana_text = diana_text[:self.typed_count]
            user_visible = ""
        elif self.phase == "USER_TYPING":
            user_visible = user_text[:self.typed_count]
        else:
            user_visible = user_text

        self._draw_diana_bubble(screen, diana_text, CHAT_RECT.y + 86)
        if user_visible:
            self._draw_user_bubble(screen, user_visible, CHAT_RECT.y + 252)

    def _draw_diana_bubble(self, screen, text, y):
        avatar_x = CHAT_RECT.x + 24
        screen.blit(self.diana_avatar, (avatar_x, y))
        bubble = pygame.Rect(avatar_x + self.avatar_size + 16, y + 4, 430, 104)
        draw_round_rect(screen, bubble, (238, 244, 252), 18)
        draw_wrapped_text(screen, text, self.chat_font, (16, 30, 48), bubble.inflate(-28, -24))

    def _draw_user_bubble(self, screen, text, y):
        avatar_x = CHAT_RECT.right - 24 - self.avatar_size
        screen.blit(self.user_avatar, (avatar_x, y))
        bubble = pygame.Rect(CHAT_RECT.x + 78, y + 4, 430, 126)
        draw_round_rect(screen, bubble, (31, 54, 76), 18, (88, 170, 205), 1)
        progress_rect = bubble.inflate(-30, -30)
        current = self.user_char_index if self.phase == "WAIT_SIGN" else -1
        visible_completed = {i for i in self.completed_indices if i < len(text)}
        draw_progress_text(screen, text, visible_completed, current, self.user_font, progress_rect)

    def _draw_feedback(self, screen):
        if self.time < self.wrong_until:
            cross = self.big_font.render("×", True, (255, 50, 68))
            screen.blit(cross, cross.get_rect(center=(580, 594)))
        if self.time < self.correct_until or self.phase == "FINISHED":
            check = self.big_font.render("✔", True, (68, 255, 130))
            screen.blit(check, check.get_rect(center=(580, 594)))

    def _draw_finished(self, screen):
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 8, 18, 210))
        screen.blit(overlay, (0, 0))
        box = pygame.Rect(300, 188, 680, 320)
        draw_round_rect(screen, box, (12, 30, 48), 28, (80, 240, 180), 3)
        check = self.big_font.render("✔", True, (80, 255, 150))
        screen.blit(check, check.get_rect(center=(640, 255)))
        title = self.title_font.render("Great work! You finished the Diana ASL chat.", True, (235, 255, 246))
        screen.blit(title, title.get_rect(center=(640, 340)))
        errors = self.hud_font.render(f"Total errors: {self.total_errors}", True, (255, 206, 76))
        screen.blit(errors, errors.get_rect(center=(640, 392)))
        run = self.small_font.render("Run again with: python -m src.game.main", True, (160, 220, 240))
        screen.blit(run, run.get_rect(center=(640, 438)))
        quit_text = self.small_font.render("Press Q or ESC to quit.", True, (180, 205, 220))
        screen.blit(quit_text, quit_text.get_rect(center=(640, 468)))

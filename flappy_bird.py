import pygame  # type: ignore
import sys
import random
import numpy as np
from typing import List, Tuple

# -------------------- 화면/게임 상수 --------------------
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60

# 새 파라미터
BIRD_X = 0.25 * SCREEN_WIDTH
BIRD_RADIUS = 18
GRAVITY = 1000.0            # px/s^2
FLAP_VELOCITY = -350.0      # px/s (위로 음수)
MAX_FALL_SPEED = 600.0

# 파이프 파라미터
PIPE_WIDTH = 80
PIPE_GAP = 130              # ✅ 세로 통과 간격 좁게
PIPE_SPEED = 180.0          # px/s (왼쪽으로 이동)
PIPE_SPACING_PX = 280       # ✅ 수평 간격 고정
PIPE_MIN_MARGIN = 80        # 화면 위/아래 여백

# 렌더 색상
COLOR_BG = (240, 235, 220)
COLOR_PIPE = (60, 170, 90)
COLOR_PIPE_DARK = (40, 120, 60)
COLOR_BIRD = (230, 80, 80)
COLOR_TEXT = (30, 30, 30)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class FlappyBirdGame:
    """
    강화학습/사람플레이 공용 게임 클래스
    """

    def __init__(self):
        self.screen = None
        self.clock = None
        self.font = None

        self.bird_y: float = 0.0
        self.bird_vy: float = 0.0
        self.pipes: List[Tuple[float, float]] = []
        self.score: int = 0
        self.prev_pipe_pass_count: int = 0
        self.rng = random.Random()

        self._last_gap_center = SCREEN_HEIGHT * 0.5

        # RL용 고정 timestep
        self._dt_for_rl = 1.0 / FPS
        self.reset()

    # -------------------- RL API --------------------
    def reset(self):
        self.bird_y = SCREEN_HEIGHT * 0.5
        self.bird_vy = 0.0
        self.score = 0
        self.prev_pipe_pass_count = 0

        # 초기 파이프 일정 간격으로 생성
        self.pipes = []
        start_x = SCREEN_WIDTH + 200
        self._last_gap_center = SCREEN_HEIGHT * 0.5
        for i in range(5):
            gap_y = self._next_gap_center()
            self.pipes.append((start_x + i * PIPE_SPACING_PX, gap_y))

        return self.get_observation()

    def step(self, action: int):
        dt = self._dt_for_rl

        # 입력
        if action == 1:
            self.bird_vy = FLAP_VELOCITY

        # 중력
        self.bird_vy += GRAVITY * dt
        self.bird_vy = clamp(self.bird_vy, -abs(FLAP_VELOCITY), MAX_FALL_SPEED)
        self.bird_y += self.bird_vy * dt

        # 파이프 이동
        new_pipes = []
        for (x, gap_y) in self.pipes:
            x -= PIPE_SPEED * dt
            if x + PIPE_WIDTH > 0:
                new_pipes.append((x, gap_y))
        self.pipes = new_pipes

        # ✅ 위치 기반 스폰: 마지막 파이프 기준
        if len(self.pipes) > 0:
            last_x, _ = self.pipes[-1]
            if last_x <= SCREEN_WIDTH - PIPE_SPACING_PX:
                gap_y = self._next_gap_center()
                self.pipes.append((last_x + PIPE_SPACING_PX, gap_y))

        # 보상/종료 판정
        reward = 0.01
        terminated = False

        pipe_pass_count = sum(1 for (x, _) in self.pipes if (x + PIPE_WIDTH) < BIRD_X)
        newly_passed = max(0, pipe_pass_count - self.prev_pipe_pass_count)
        if newly_passed > 0:
            self.score += newly_passed
            reward += 1.0 * newly_passed
        self.prev_pipe_pass_count = pipe_pass_count

        # 충돌 체크
        if (self.bird_y - BIRD_RADIUS) < 0 or (self.bird_y + BIRD_RADIUS) > SCREEN_HEIGHT:
            terminated = True
            reward -= 1.0
        else:
            bx = BIRD_X
            by = self.bird_y
            for (x, gap_y) in self.pipes:
                top_rect = pygame.Rect(int(x), 0, PIPE_WIDTH, int(gap_y - PIPE_GAP / 2))
                bot_rect = pygame.Rect(int(x), int(gap_y + PIPE_GAP / 2), PIPE_WIDTH,
                                       SCREEN_HEIGHT - int(gap_y + PIPE_GAP / 2))
                if self._circle_rect_collision(bx, by, BIRD_RADIUS, top_rect) or \
                   self._circle_rect_collision(bx, by, BIRD_RADIUS, bot_rect):
                    terminated = True
                    reward -= 1.0
                    break

        return self.get_observation(), reward, terminated

    def get_observation(self) -> np.ndarray:
        next_pipe = None
        min_dx = float("inf")
        for (x, gap_y) in self.pipes:
            dx = x + PIPE_WIDTH - BIRD_X
            if dx >= 0 and dx < min_dx:
                min_dx = dx
                next_pipe = (x, gap_y)

        if next_pipe is None:
            x = float(SCREEN_WIDTH)
            gap_y = self._next_gap_center()
        else:
            x, gap_y = next_pipe

        bird_y_norm = clamp(self.bird_y / SCREEN_HEIGHT, 0.0, 1.0)
        vy_norm = clamp((self.bird_vy + MAX_FALL_SPEED) / (2 * MAX_FALL_SPEED), 0.0, 1.0)
        pipe_right_x = x + PIPE_WIDTH
        next_pipe_x_norm = clamp(pipe_right_x / SCREEN_WIDTH, 0.0, 1.0)
        next_pipe_gap_y_norm = clamp(gap_y / SCREEN_HEIGHT, 0.0, 1.0)
        gap_top_norm = clamp((gap_y - PIPE_GAP / 2) / SCREEN_HEIGHT, 0.0, 1.0)
        gap_bottom_norm = clamp((gap_y + PIPE_GAP / 2) / SCREEN_HEIGHT, 0.0, 1.0)

        return np.array(
            [bird_y_norm, vy_norm, next_pipe_x_norm, next_pipe_gap_y_norm,
             gap_top_norm, gap_bottom_norm],
            dtype=np.float32
        )

    # -------------------- 내부 --------------------
    def _bounds_for_gap_center(self):
        lo = PIPE_MIN_MARGIN + PIPE_GAP / 2
        hi = SCREEN_HEIGHT - PIPE_MIN_MARGIN - PIPE_GAP / 2
        return lo, hi

    def _next_gap_center(self) -> float:
        lo, hi = self._bounds_for_gap_center()
        # ✅ 완전 랜덤
        self._last_gap_center = self.rng.uniform(lo, hi)
        return self._last_gap_center

    @staticmethod
    def _circle_rect_collision(cx: float, cy: float, cr: float, rect: pygame.Rect) -> bool:
        closest_x = clamp(cx, rect.left, rect.right)
        closest_y = clamp(cy, rect.top, rect.bottom)
        dx = cx - closest_x
        dy = cy - closest_y
        return (dx * dx + dy * dy) <= (cr * cr)

    # -------------------- 렌더링 --------------------
    def render(self, surface):
        surface.fill(COLOR_BG)
        for (x, gap_y) in self.pipes:
            top_h = int(gap_y - PIPE_GAP / 2)
            bot_y = int(gap_y + PIPE_GAP / 2)
            bot_h = SCREEN_HEIGHT - bot_y

            top_rect = pygame.Rect(int(x), 0, PIPE_WIDTH, top_h)
            bot_rect = pygame.Rect(int(x), bot_y, PIPE_WIDTH, bot_h)
            pygame.draw.rect(surface, COLOR_PIPE, top_rect, border_radius=8)
            pygame.draw.rect(surface, COLOR_PIPE, bot_rect, border_radius=8)

            rim_thickness = 8
            rim_top = pygame.Rect(int(x), top_h - rim_thickness, PIPE_WIDTH, rim_thickness)
            rim_bot = pygame.Rect(int(x), bot_y, PIPE_WIDTH, rim_thickness)
            pygame.draw.rect(surface, COLOR_PIPE_DARK, rim_top, border_radius=4)
            pygame.draw.rect(surface, COLOR_PIPE_DARK, rim_bot, border_radius=4)

        pygame.draw.circle(surface, COLOR_BIRD, (int(BIRD_X), int(self.bird_y)), BIRD_RADIUS)

        if self.font is not None:
            txt = self.font.render(f"Score: {self.score}", True, COLOR_TEXT)
            surface.blit(txt, (12, 10))

    # -------------------- 사람 플레이 --------------------
    def run_for_human(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Flappy Bird - RL/Manual")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)

        running = True
        alive = True

        while running:
            dt = self.clock.tick(FPS) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    if event.key == pygame.K_r:
                        self.reset()
                        alive = True
                    if alive and event.key in (pygame.K_SPACE, pygame.K_UP, pygame.K_w):
                        self.bird_vy = FLAP_VELOCITY

            if alive:
                # step 로직 그대로 적용
                self.bird_vy += GRAVITY * dt
                self.bird_vy = clamp(self.bird_vy, -abs(FLAP_VELOCITY), MAX_FALL_SPEED)
                self.bird_y += self.bird_vy * dt

                new_pipes = []
                for (x, gap_y) in self.pipes:
                    x -= PIPE_SPEED * dt
                    if x + PIPE_WIDTH > 0:
                        new_pipes.append((x, gap_y))
                self.pipes = new_pipes

                if len(self.pipes) > 0:
                    last_x, _ = self.pipes[-1]
                    if last_x <= SCREEN_WIDTH - PIPE_SPACING_PX:
                        gap_y = self._next_gap_center()
                        self.pipes.append((last_x + PIPE_SPACING_PX, gap_y))

                pipe_pass_count = sum(1 for (x, _) in self.pipes if (x + PIPE_WIDTH) < BIRD_X)
                newly_passed = max(0, pipe_pass_count - self.prev_pipe_pass_count)
                if newly_passed > 0:
                    self.score += newly_passed
                self.prev_pipe_pass_count = pipe_pass_count

                if (self.bird_y - BIRD_RADIUS) < 0 or (self.bird_y + BIRD_RADIUS) > SCREEN_HEIGHT:
                    alive = False
                else:
                    bx, by = BIRD_X, self.bird_y
                    for (x, gap_y) in self.pipes:
                        top_rect = pygame.Rect(int(x), 0, PIPE_WIDTH, int(gap_y - PIPE_GAP / 2))
                        bot_rect = pygame.Rect(int(x), int(gap_y + PIPE_GAP / 2), PIPE_WIDTH,
                                               SCREEN_HEIGHT - int(gap_y + PIPE_GAP / 2))
                        if self._circle_rect_collision(bx, by, BIRD_RADIUS, top_rect) or \
                           self._circle_rect_collision(bx, by, BIRD_RADIUS, bot_rect):
                            alive = False
                            break

            self.render(self.screen)
            if not alive and self.font is not None:
                msg = self.font.render("Game Over - Press R to Restart", True, (180, 20, 20))
                self.screen.blit(msg, (SCREEN_WIDTH // 2 - msg.get_width() // 2, SCREEN_HEIGHT // 2 - 20))

            pygame.display.flip()

        pygame.quit()


if __name__ == "__main__":
    game = FlappyBirdGame()
    game.run_for_human()

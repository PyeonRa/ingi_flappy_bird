import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame  # type: ignore
from flappy_bird import FlappyBirdGame, SCREEN_WIDTH, SCREEN_HEIGHT

class FlappyBirdEnv(gym.Env):
    """
    Gymnasium 환경 래퍼
    - action_space: Discrete(2)  (0: no-op, 1: flap)
    - observation_space: Box(6,)  [0,1] 정규화 연속값
    """
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=None, max_steps: int = 5000):
        super().__init__()
        self.game = FlappyBirdGame()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.steps_taken = 0

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(6,), dtype=np.float32
        )

        self._surface = None

    def step(self, action):
        obs, reward, terminated = self.game.step(int(action))
        self.steps_taken += 1
        truncated = self.steps_taken >= self.max_steps

        if self.render_mode == "human":
            self._ensure_surface()
            self.game.render(self._surface)
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            surf = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.game.render(surf)
            obs_rgb = pygame.surfarray.array3d(surf).swapaxes(0, 1)
            return obs, reward, terminated, truncated, {"rgb_array": obs_rgb}

        return obs, reward, terminated, truncated, {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps_taken = 0
        obs = self.game.reset()

        if self.render_mode == "human":
            self._ensure_surface()
            self.game.render(self._surface)
            pygame.display.flip()

        return obs, {}

    def render(self):
        if self.render_mode == "human":
            self._ensure_surface()
            self.game.render(self._surface)
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            surf = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.game.render(surf)
            return pygame.surfarray.array3d(surf).swapaxes(0, 1)

    def close(self):
        try:
            pygame.quit()  # type: ignore
        except Exception:
            pass

    def _ensure_surface(self):
        if self._surface is None:
            pygame.init()  # type: ignore
            self._surface = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Flappy Bird - Gymnasium Env")
            self.game.font = pygame.font.Font(None, 36)

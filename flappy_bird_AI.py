import os
import time
import pygame
from stable_baselines3 import PPO

# 로컬 파일에서 환경 가져오기
from flappy_bird_env import FlappyBirdEnv

def main():
    model_path = os.path.abspath("ingi/flappy_model.zip")  # 모델 경로 (필요시 수정)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

    # 렌더링 모드: 'human'이면 실제 창으로 보면서 플레이 진행
    env = FlappyBirdEnv(render_mode="human")

    # 모델 로드 (env를 반드시 넣어줘야 predict/learn OK)
    model = PPO.load(model_path, env=env)

    # 한 에피소드 플레이
    obs, info = env.reset()
    done, trunc = False, False
    episode_return = 0.0

    clock = pygame.time.Clock()

    while not (done or trunc):
        # 학습된 정책으로 행동 선택
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = env.step(action)
        episode_return += reward

        # FPS 제한(너무 빨리 지나가면 보기 힘드니 살짝 제한)
        clock.tick(60)

    print(f"Episode return: {episode_return:.3f}")
    env.close()

if __name__ == "__main__":
    main()

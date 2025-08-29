import os
import sys
import time
import argparse
import pygame
from typing import Optional

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback

# 현재 파일 기준으로 같은 폴더(ingi) 안의 모듈을 import 가능하게
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from flappy_bird_env import FlappyBirdEnv  # 같은 폴더에 있어야 함


def build_vec_env(n_envs: int = 1):
    def make_env():
        return FlappyBirdEnv(render_mode=None)
    if n_envs < 1:
        n_envs = 1
    vec = DummyVecEnv([make_env for _ in range(n_envs)])
    vec = VecMonitor(vec)
    return vec


class LiveRenderCallback(BaseCallback):
    """
    학습 중 일정 스텝마다 현재 정책을 'human' 렌더 모드로 재생해 보여주는 콜백.
    - every_steps: 몇 학습 스텝마다 재생할지 (0이면 비활성)
    - max_steps: 재생 시 최대 스텝 수 (에피소드가 먼저 끝나면 중단)
    - fps: 재생 시 FPS
    """
    def __init__(self, every_steps: int = 10000, max_steps: int = 2000, fps: int = 60, verbose: int = 0):
        super().__init__(verbose)
        self.every_steps = max(0, int(every_steps))
        self.max_steps = max(1, int(max_steps))
        self.fps = max(1, int(fps))
        self._next_render_at = self.every_steps if self.every_steps > 0 else None

    def _on_step(self) -> bool:
        if not self.training_env or not self.model:
            return True
        if self.every_steps <= 0:
            return True

        # 현재 학습 스텝
        n_steps = self.num_timesteps
        if self._next_render_at is not None and n_steps >= self._next_render_at:
            try:
                if self.verbose:
                    print(f"\n👀 LiveRender: timesteps={n_steps} ... 창 재생 시작")
                self._play_once()
            except Exception as e:
                print(f"⚠️ LiveRender 에러: {e}")
            finally:
                # 다음 예약
                self._next_render_at += self.every_steps
        return True

    def _play_once(self):
        env_human = FlappyBirdEnv(render_mode="human")
        obs, info = env_human.reset()
        done, trunc = False, False
        clock = pygame.time.Clock()
        steps = 0

        while not (done or trunc) and steps < self.max_steps:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, trunc, info = env_human.step(action)
            steps += 1
            clock.tick(self.fps)

        env_human.close()
        if self.verbose:
            print("✅ LiveRender 종료")


def eval_one_episode(model: PPO, fps: int = 60) -> float:
    """
    수동 평가: 1에피소드만 렌더 모드로 플레이 (원하면 main에서 호출)
    """
    env_human = FlappyBirdEnv(render_mode="human")
    obs, info = env_human.reset()
    done, trunc = False, False
    ep_ret = 0.0
    clock = pygame.time.Clock()

    try:
        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, trunc, info = env_human.step(action)
            ep_ret += float(reward)
            clock.tick(fps)
    finally:
        env_human.close()

    return ep_ret


def main():
    parser = argparse.ArgumentParser(description="FlappyBird - Local Continuous Trainer (with Live Render)")
    parser.add_argument("--model_path", type=str, default=os.path.join(BASE_DIR, "flappy_model"),
                        help="모델 저장 경로(확장자 없이). 예: ingi/flappy_model")
    parser.add_argument("--log_dir", type=str, default=os.path.join(BASE_DIR, "logs"),
                        help="TensorBoard 로그 디렉토리")
    parser.add_argument("--tb", type=int, default=0, choices=[0, 1],
                        help="TensorBoard 로깅 사용(1) / 미사용(0). 미설치면 0 권장")
    parser.add_argument("--chunk_steps", type=int, default=50000,
                        help="한 번에 학습할 스텝 수")
    parser.add_argument("--save_every_chunks", type=int, default=2,
                        help="몇 청크마다 저장할지 (예: 2 -> 100k 스텝마다 저장)")
    parser.add_argument("--n_envs", type=int, default=1,
                        help="병렬 환경 개수 (CPU 여유 있으면 2~4 권장)")
    parser.add_argument("--eval_every_chunks", type=int, default=0,
                        help="몇 청크마다 수동 평가(1에피소드)를 보여줄지 (0이면 끔)")
    parser.add_argument("--eval_fps", type=int, default=60,
                        help="수동 평가 렌더 FPS")
    parser.add_argument("--live_render_every_steps", type=int, default=10000,
                        help="학습 중 라이브 렌더 주기(스텝). 0이면 비활성")
    parser.add_argument("--live_render_max_steps", type=int, default=1500,
                        help="라이브 렌더 재생 최대 스텝(에피소드가 먼저 끝나면 중단)")
    parser.add_argument("--live_render_fps", type=int, default=60,
                        help="라이브 렌더 FPS")
    args = parser.parse_args()

    model_path = args.model_path
    log_dir = args.log_dir
    use_tb = bool(args.tb)
    chunk_steps = args.chunk_steps
    save_every = max(1, args.save_every_chunks)
    n_envs = args.n_envs
    eval_every = max(0, args.eval_every_chunks)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if use_tb:
        os.makedirs(log_dir, exist_ok=True)

    # 벡터라이즈 환경
    vec_env = build_vec_env(n_envs=n_envs)

    # 기존 모델이 있으면 로드, 없으면 새로 생성
    if os.path.exists(model_path + ".zip"):
        print(f"🔁 기존 모델 로드: {model_path}.zip")
        if use_tb:
            model = PPO.load(model_path, env=vec_env, verbose=1, tensorboard_log=log_dir)
        else:
            model = PPO.load(model_path, env=vec_env, verbose=1)
    else:
        print("🆕 새 모델 생성")
        if use_tb:
            model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=log_dir)
        else:
            model = PPO("MlpPolicy", vec_env, verbose=1)

    # 라이브 렌더 콜백 설정
    live_cb = LiveRenderCallback(
        every_steps=args.live_render_every_steps,
        max_steps=args.live_render_max_steps,
        fps=args.live_render_fps,
        verbose=1
    )

    chunk_count = 0
    total_steps_done = 0

    try:
        while True:
            t0 = time.time()
            # 이어 학습 (+ 라이브 렌더 콜백)
            model.learn(total_timesteps=chunk_steps, reset_num_timesteps=False, callback=live_cb)
            chunk_count += 1
            total_steps_done += chunk_steps
            dt = time.time() - t0
            print(f"✅ chunk {chunk_count} 완료 (+{chunk_steps} steps, 누적 {total_steps_done}) - {dt:.1f}s")

            # 주기 저장
            if (chunk_count % save_every) == 0:
                model.save(model_path)
                print(f"💾 자동 저장: {model_path}.zip")

            # 주기 수동 평가(옵션)
            if eval_every > 0 and (chunk_count % eval_every) == 0:
                try:
                    print("🧪 평가 1에피소드 재생…")
                    ep_ret = eval_one_episode(model, fps=args.eval_fps)
                    print(f"🎯 평가 episode return: {ep_ret:.3f}")
                except Exception as e:
                    print(f"⚠️ 평가 중 오류: {e}")

    except KeyboardInterrupt:
        print("🛑 학습 중단(Ctrl+C). 마지막 상태 저장…")
        model.save(model_path)
        print(f"✅ 마지막 저장 완료: {model_path}.zip")

    except Exception as e:
        print(f"⚠️ 예외 발생: {e}")
        print("현재 상태 저장 시도…")
        try:
            model.save(model_path)
            print(f"✅ 비정상 종료 저장 완료: {model_path}.zip")
        except Exception as e2:
            print(f"❌ 저장 실패: {e2}")

    finally:
        vec_env.close()


if __name__ == "__main__":
    main()

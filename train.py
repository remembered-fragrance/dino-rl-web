"""Train PPO (Stable-Baselines3) on the self-hosted game, one headless browser per env.

By default the game runs in lockstep (faster than real time) and half the games start at a random speed.

    python train.py --envs 8 --steps 2000000
    python train.py --resume runs/<run>/final.zip --steps 100000
    tensorboard --logdir runs
"""

import argparse
import datetime
from functools import partial
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from dino_env import DinoEnv


class RealTimeGameCallback(BaseCallback):
    """The game runs in real time: freeze every browser while PPO updates the network. Also logs game scores."""

    def _on_rollout_end(self):
        self.training_env.env_method("pause")

    def _on_rollout_start(self):
        self.training_env.env_method("resume")

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "episode" in info:
                self.logger.record_mean("rollout/score", info["episode"]["score"])
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--envs", type=int, default=8, help="parallel headless browsers")
    p.add_argument("--steps", type=int, default=200_000, help="total env steps")
    p.add_argument("--step-ms", type=int, default=50, help="agent decision interval")
    p.add_argument("--resume", help="continue from a saved model .zip")
    p.add_argument("--realtime", action="store_true", help="train in real time instead of lockstep (slower)")
    p.add_argument("--max-steps", type=int, default=3000,
                   help="cut training games here; without it long top-speed games crowd out slow starts and the "
                        "policy forgets them (v6 lost ~1/3 of games at the first cactus)")
    p.add_argument("--lr", type=float, default=3e-4, help="learning rate (also overrides a resumed model's)")
    a = p.parse_args()

    run = Path("runs") / datetime.datetime.now().strftime("ppo-%m%d-%H%M")
    run.mkdir(parents=True)
    (run / "step_ms.txt").write_text(str(a.step_ms))  # evaluate.py plays the model at this interval
    # Subprocesses, not DummyVecEnv: each env runs its own browser (and in real time sleeps to keep its pace).
    make_env = partial(
        DinoEnv, site="local", step_ms=a.step_ms, max_steps=a.max_steps,
        lockstep=not a.realtime, random_start_speed=not a.realtime,
    )
    env = VecMonitor(
        SubprocVecEnv([make_env] * a.envs),
        str(run / "train"),
        info_keywords=("score",),
    )
    if a.resume:
        model = PPO.load(a.resume, env, device="cpu", tensorboard_log=str(run), learning_rate=a.lr)
    else:
        model = PPO(
            "MlpPolicy", env, n_steps=512, batch_size=256, n_epochs=10, gamma=0.99, gae_lambda=0.95,
            ent_coef=0.01, learning_rate=a.lr, device="cpu", tensorboard_log=str(run), verbose=1,
        )

    checkpoints = CheckpointCallback(max(100_000 // a.envs, 1), str(run), name_prefix="ppo")
    try:
        model.learn(a.steps, callback=[RealTimeGameCallback(), checkpoints], reset_num_timesteps=not a.resume)
    finally:
        model.save(run / "final")
        env.close()
        print(f"--> saved {run / 'final.zip'}")


if __name__ == "__main__":
    main()

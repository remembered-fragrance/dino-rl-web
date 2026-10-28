"""Play N games with a policy and print the score summary.

    python evaluate.py --agent rule --site local --episodes 5
    python evaluate.py --agent rule --site online --headed
"""

import argparse
from pathlib import Path

import numpy as np

from baselines.rule_bot import rule_policy
from dino_env import NOOP, DinoEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", choices=["rule", "random", "noop", "ppo"], default="rule")
    p.add_argument("--model", help="ppo: path to a saved model .zip")
    p.add_argument("--site", default="local", help="local | online | any URL serving the game")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--headed", action="store_true", help="show the browser window")
    p.add_argument("--max-steps", type=int, default=None, help="cut a game after this many steps")
    p.add_argument("--stop-at", type=int, default=None, help="end a game once its score reaches this")
    p.add_argument("--lockstep", action="store_true", help="local only: run the game faster than real time")
    p.add_argument("--lead", type=float, default=14, help="rule bot: act when obstacle is closer than lead*speed px")
    p.add_argument("--step-ms", type=int, default=None, help="decision interval (default: the model's, else 50)")
    a = p.parse_args()
    step_ms = a.step_ms or 50
    if a.agent == "ppo":
        if not a.model:
            p.error("--agent ppo needs --model")
        from stable_baselines3 import PPO  # heavy import, only when needed

        model = PPO.load(a.model, device="cpu")
        # A policy only works at the decision interval it was trained with; train.py records it next to the model.
        saved = Path(a.model).with_name("step_ms.txt")
        step_ms = a.step_ms or (int(saved.read_text()) if saved.exists() else 50)

    env = DinoEnv(site=a.site, headless=not a.headed, max_steps=a.max_steps, lockstep=a.lockstep, step_ms=step_ms)
    policy = {
        "rule": lambda o: rule_policy(o, a.lead),
        "random": lambda o: env.action_space.sample(),
        "noop": lambda o: NOOP,
        "ppo": lambda o: int(model.predict(o, deterministic=True)[0]),
    }[a.agent]

    scores = []
    try:
        for ep in range(a.episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                obs, _, terminated, truncated, info = env.step(policy(obs))
                done = terminated or truncated or (a.stop_at is not None and info["score"] >= a.stop_at)
            scores.append(info["score"])
            print(f"game {ep + 1}: score {info['score']}" + ("" if terminated else " (stopped alive)"))
    finally:
        env.close()
    print(f"{a.agent} @ {a.site}: mean {np.mean(scores):.0f} ± {np.std(scores):.0f}, max {max(scores)} over {len(scores)} games")
    if a.stop_at is not None:
        print(f"reached {a.stop_at}: {sum(s >= a.stop_at for s in scores)}/{len(scores)} games")


if __name__ == "__main__":
    main()

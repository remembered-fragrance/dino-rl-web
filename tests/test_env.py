import time

import pytest

from baselines.rule_bot import rule_policy
from dino_env import DUCK, JUMP, NOOP, DinoEnv, features
from dino_env.env import OBS_SIZE

SNAP = {
    "crashed": False, "score": 0, "speed": 6.5, "x": 50, "y": 93, "ground": 93, "jumping": False, "ducking": False,
    "obstacles": [{"x": 130, "y": 105, "w": 17, "h": 35, "bird": False}],
}


def test_features():
    obs = features(SNAP)
    assert obs.shape == (OBS_SIZE,)
    assert obs[4] == pytest.approx(80 / 600)
    assert obs[9] == 1.5  # missing second obstacle reads as far away
    assert obs[-1] == pytest.approx(80 / 6.5 / 30)  # half-seconds until the cactus arrives


def test_rule_bot():
    assert rule_policy(features(SNAP)) == JUMP  # cactus 80 px away < 14 * 6.5 = 91
    assert rule_policy(features({**SNAP, "obstacles": []})) == NOOP
    airborne = {**SNAP, "jumping": True, "y": 40}
    assert rule_policy(features(airborne)) == JUMP  # obstacle still near: stay up
    assert rule_policy(features({**airborne, "obstacles": []})) == DUCK  # cleared: fast-drop
    assert rule_policy(features({**airborne, "ducking": True})) == DUCK  # never cancel a drop
    bird = {"x": 100, "w": 46, "h": 40, "bird": True}
    assert rule_policy(features({**SNAP, "obstacles": [{**bird, "y": 75}]})) == DUCK
    assert rule_policy(features({**SNAP, "obstacles": [{**bird, "y": 50}]})) == NOOP


def test_local_game_crashes_and_restarts():
    env = DinoEnv(site="local")
    try:
        for _ in range(2):  # second round checks restart after a crash
            obs, _ = env.reset()
            assert obs.shape == (OBS_SIZE,)
            for _ in range(400):  # ~20 s; standing still must hit the first cactus
                obs, reward, terminated, _, info = env.step(NOOP)
                if terminated:
                    break
            assert terminated and reward == -10.0 and info["score"] > 0
    finally:
        env.close()


def test_local_pause_resume_and_reset_mid_game():
    env = DinoEnv(site="local")
    distance = lambda: env.page.evaluate("() => Runner.instance_.distanceRan")
    try:
        env.reset()
        for _ in range(50):  # ~2.5 s: intro + start, still before the first obstacle
            env.step(NOOP)
        env.pause()
        frozen = distance()
        time.sleep(0.5)
        assert distance() == frozen
        env.resume()
        env.resume()  # a second resume must not start a second frame loop
        time.sleep(0.5)
        moved = distance() - frozen
        # 0.5 s at speed ~6 px/frame, 60 fps ≈ 180 px; a doubled loop would give ~360
        assert 100 < moved < 260
        env.reset()  # the game is still running: reset must really start a new one
        assert distance() < 50
    finally:
        env.close()


def test_local_lockstep_is_faster_than_real_time():
    env = DinoEnv(site="local", lockstep=True)
    try:
        for _ in range(2):  # second round checks restart in lockstep
            env.reset()
            start, steps = time.perf_counter(), 0
            terminated = False
            while not terminated and steps < 2000:
                *_, terminated, _, info = env.step(NOOP)
                steps += 1
            assert terminated and info["score"] > 0
        # Real time is 50 ms per step; lockstep must be several times faster.
        assert (time.perf_counter() - start) / steps < 0.05 / 3
    finally:
        env.close()


def test_local_letting_go_of_jump_mid_air_drops_fast():
    env = DinoEnv(site="local")
    try:
        env.reset()
        for _ in range(70):  # ~3.5 s: let the intro jump land, still before the first obstacle
            env.step(NOOP)
        env.step(JUMP)
        env.step(JUMP)
        assert env._last["jumping"] and not env._last["ducking"]
        env.step(NOOP)
        assert env._last["ducking"]  # snapshot "ducking" includes the game's speedDrop flag
    finally:
        env.close()

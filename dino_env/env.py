"""Gymnasium env: plays the Chrome Dino game in a real browser through Playwright."""

import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from playwright.sync_api import sync_playwright

SITES = {
    "local": (Path(__file__).resolve().parent.parent / "game" / "index.html").as_uri(),
    "online": "https://chromedino.com/embed/",
}

NOOP, JUMP, DUCK = 0, 1, 2
N_OBSTACLES = 2
OBS_SIZE = 4 + 5 * N_OBSTACLES + 1

# One round trip per step. Obstacles the dino has fully passed are dropped.
SNAPSHOT_JS = """() => {
  const r = Runner.instance_, t = r.tRex;
  return {
    crashed: r.crashed, score: Math.floor(r.distanceRan * 0.025), speed: r.currentSpeed,
    x: t.xPos, y: t.yPos, ground: t.groundYPos, jumping: t.jumping, ducking: t.ducking || t.speedDrop,
    obstacles: r.horizon.obstacles.filter(o => o.xPos + o.width > t.xPos).slice(0, %d).map(o => ({
      x: o.xPos, y: o.yPos, w: o.width, h: o.typeConfig.height, bird: o.typeConfig.type === 'PTERODACTYL'})),
  };
}""" % N_OBSTACLES

# The game is not frame-rate independent (speed grows per frame, jump positions round per frame), and a visible
# Chrome follows the monitor (301 fps here) while training ran headless at 60. Cap requestAnimationFrame at an
# average 60 fps; ids stay stable so cancelAnimationFrame (used by stop/restart) still cancels a pending frame.
CAP_60_FPS_JS = """(() => {
  const raf = window.requestAnimationFrame.bind(window), caf = window.cancelAnimationFrame.bind(window);
  const P = 1000 / 60, live = new Map();
  let next = 1, due = 0, frame = -1;
  window.requestAnimationFrame = cb => {
    const id = next++;
    const tick = t => {
      if (!live.has(id)) return;
      // Gate whole display frames (all callbacks of a frame share t), or a second rAF user starves the game loop.
      if (t !== frame) {
        if (t < due - 1) { live.set(id, raf(tick)); return; }
        frame = t;
        due = t > due + P ? t + P : due + P;  // resync after a stall instead of bursting
      }
      live.delete(id);
      cb(t);
    };
    live.set(id, raf(tick));
    return id;
  };
  window.cancelAnimationFrame = id => { if (live.has(id)) caf(live.get(id)); live.delete(id); };
})();"""

# Lockstep (for training): step() drives the page clock and the frame loop, so the game runs as fast as the CPU
# allows with exact 60 fps frames. getTimeStamp() in the game reads performance.now().
LOCKSTEP_JS = """(() => {
  let vt = performance.now();
  performance.now = () => vt;
  window.requestAnimationFrame = () => 1;  // the frame loop only advances through __advance()
  window.cancelAnimationFrame = () => {};
  window.__advance = frames => { for (let i = 0; i < frames; i++) { vt += 1000 / 60; Runner.instance_.update(); } };
})();"""
STEP_LOCKSTEP_JS = "(frames) => { window.__advance(frames); return (%s)(); }" % SNAPSHOT_JS

# The game's restart() is a no-op while a game is running (e.g. one cut at max_steps): stop it first.
RESTART_JS = "() => { const r = Runner.instance_; r.stop(); r.paused = false; r.restart(); }"
# The game's play() resets the dino to RUNNING, which turns an in-progress jump into slow motion,
# so restart the frame loop by hand. Only when stopped: a second loop would double the game speed.
RESUME_JS = """() => { const r = Runner.instance_;
  if (!r.raqId && !r.crashed) { r.playing = true; r.paused = false; r.time = performance.now(); r.update(); } }"""


def features(s: dict) -> np.ndarray:
    """Snapshot -> normalised vector: [speed, height, jumping, ducking/fast-dropping]
    + per obstacle [dist, top, w, h, bird] + [time until the nearest obstacle arrives, in half-seconds]."""
    obs = np.zeros(OBS_SIZE, np.float32)
    obs[:4] = s["speed"] / 13, (s["ground"] - s["y"]) / 100, s["jumping"], s["ducking"]
    for i in range(N_OBSTACLES):
        o = s["obstacles"][i] if i < len(s["obstacles"]) else None
        # A missing obstacle looks like one far off-screen.
        obs[4 + 5 * i : 9 + 5 * i] = (
            ((o["x"] - s["x"]) / 600, o["y"] / 150, o["w"] / 100, o["h"] / 100, o["bird"]) if o else (1.5, 0, 0, 0, 0)
        )
    # Obstacles move `speed` px per frame (60 fps): jump timing depends on dist/speed, not dist alone.
    obs[-1] = min(obs[4] * 600 / max(s["speed"], 1) / 30, 5)
    return obs


class DinoEnv(gym.Env):
    """Actions: 0 run (mid-air: drop fast), 1 jump (holds Up), 2 duck (holds Down; mid-air: drop fast).

    Reward: +0.1 per step alive, -jump_cost per jump started, -10 on crash. Without the jump cost PPO
    settles on hopping non-stop (~55 points). The game runs in real time; the agent acts every `step_ms`.
    """

    def __init__(
        self, site="local", headless=True, step_ms=50, max_steps=None, jump_cost=0.5,
        lockstep=False, random_start_speed=False,
    ):
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(-5, 5, (OBS_SIZE,), np.float32)
        self.step_s = step_ms / 1000
        self.max_steps = max_steps
        self.jump_cost = jump_cost
        self.lockstep = lockstep
        self.frames = round(step_ms * 60 / 1000)
        # Half the games start at a random speed, to practise the top speed where a long game spends most time.
        self.random_start_speed = random_start_speed

        self._pw = sync_playwright().start()
        # Visible runs use the installed Google Chrome: Playwright's bundled chrome.exe fails to start on
        # this machine (WinError 14001, missing VC++ runtime), and a demo in real Chrome reads better anyway.
        self._browser = self._pw.chromium.launch(headless=headless, channel=None if headless else "chrome")
        self.page = self._browser.new_page(viewport={"width": 900, "height": 400})
        # Never let the agent's scores reach chromedino.com's public leaderboard.
        self.page.route("**/inc/**", lambda route: route.abort())
        self.page.add_init_script(LOCKSTEP_JS if lockstep else CAP_60_FPS_JS)
        self.page.goto(SITES.get(site, site))
        self.page.wait_for_function("() => typeof Runner !== 'undefined' && !!Runner.instance_")
        self._started = False
        self._held = {"ArrowUp": False, "ArrowDown": False}

    def _snapshot(self) -> dict:
        return self.page.evaluate(SNAPSHOT_JS)

    def _press(self, action: int, jumping: bool):
        kb = self.page.keyboard
        want_up = action == JUMP
        if want_up and self._held["ArrowUp"] and not jumping:
            # Landed while still holding Up: the game only jumps on a fresh keydown.
            kb.up("ArrowUp")
            self._held["ArrowUp"] = False
        # Mid-air, letting go of Jump drops fast (holds Down). PPO never found fast-drop as a rare separate
        # action, and needs it when obstacles come close together.
        want_down = action == DUCK or (jumping and not want_up)
        for key, want in (("ArrowUp", want_up), ("ArrowDown", want_down)):
            if want != self._held[key]:
                (kb.down if want else kb.up)(key)
                self._held[key] = want

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._press(NOOP, jumping=False)
        if self._started:
            self.page.evaluate(RESTART_JS)
        else:
            self.page.keyboard.press("Space")  # first game on this page: Space starts it
            self._started = True
        if self.random_start_speed and self.np_random.random() < 0.5:
            self.page.evaluate("(s) => Runner.instance_.setSpeed(s)", float(self.np_random.uniform(6, 13)))
        self._steps = 0
        self._next_t = time.perf_counter() + self.step_s
        self._last = self._snapshot()
        return features(self._last), {"score": self._last["score"]}

    def step(self, action):
        jumped = int(action) == JUMP and not self._last["jumping"]
        self._press(int(action), self._last["jumping"])
        if self.lockstep:
            # A real-time 50 ms step spans 2-4 frames; randomise so the policy survives that jitter live.
            frames = self.frames + int(self.np_random.choice([-1, 0, 1], p=[0.1, 0.8, 0.1]))
            self._last = s = self.page.evaluate(STEP_LOCKSTEP_JS, frames)
        else:
            # Keep a fixed decision rate no matter how long the browser round trip took.
            delay = self._next_t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            self._next_t = max(self._next_t + self.step_s, time.perf_counter())
            self._last = s = self._snapshot()
        self._steps += 1
        crashed = bool(s["crashed"])
        truncated = self.max_steps is not None and self._steps >= self.max_steps
        reward = -10.0 if crashed else 0.1 - (self.jump_cost if jumped else 0.0)
        return features(s), reward, crashed, truncated, {"score": s["score"]}

    def pause(self):
        """Freeze the game, e.g. while PPO updates the network (otherwise it keeps running in real time)."""
        if not self.lockstep:  # a lockstep game only moves inside step()
            self.page.evaluate("() => Runner.instance_.stop()")

    def resume(self):
        if not self.lockstep:
            self.page.evaluate(RESUME_JS)
        self._next_t = time.perf_counter() + self.step_s

    def close(self):
        self._browser.close()
        self._pw.stop()

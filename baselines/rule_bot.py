"""Hand-written baseline: jump over cacti and low birds, duck under mid-height birds.

Uses the same observation vector as the RL agent, so the comparison is fair.
"""

from dino_env import DUCK, JUMP, NOOP


def rule_policy(obs, lead_frames: float = 14) -> int:
    speed, jumping, down = obs[0] * 13, obs[2], obs[3]
    dist, top, bird = obs[4] * 600, obs[5] * 150, obs[8]
    # Obstacles move `speed` px per frame at 60 fps: act when one is `lead_frames` away.
    near = dist <= lead_frames * speed
    if jumping:
        # Mid-air the only choice is fast-dropping. Drop once the obstacle jumped for is cleared,
        # and keep dropping: releasing Down cancels the drop.
        return DUCK if down or not near else JUMP
    if not near or (bird and top < 60):  # high bird (y=50) flies over a standing dino
        return NOOP
    if bird and top < 90:  # mid bird (y=75): duck under it
        return DUCK
    return JUMP

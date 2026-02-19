import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.pool import binding


class Pool(pufferlib.PufferEnv):
    def __init__(
        self,
        num_envs=256,
        width=2.84,
        height=1.42,
        ball_radius=0.03,
        pocket_radius=0.06,
        friction=0.992,
        restitution=0.96,
        impulse=0.12,
        min_power=0.35,
        reward_step=0.0,
        reward_shot=-0.001,
        reward_legal_pot=0.03,
        reward_illegal_pot=-0.015,
        reward_foul=-0.05,
        reward_win=1.0,
        fast_forward=True,
        max_physics_steps=96,
        max_steps=200,
        report_interval=128,
        render_mode='human',
        buf=None,
        seed=0,
    ):
        self.render_mode = render_mode
        self.num_tables = num_envs
        self.num_agents = num_envs * 2
        self.report_interval = report_interval
        self.tick = 0

        self.single_observation_space = gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(89,),
            dtype=np.float32,
        )
        # Action[0]: shot direction in [0..15] around 360 degrees
        # Action[1]: power bin in [0..4]
        self.single_action_space = gymnasium.spaces.MultiDiscrete(
            [16, 5], dtype=np.int32
        )

        super().__init__(buf)

        c_envs = []
        for i in range(num_envs):
            start = 2 * i
            end = start + 2
            env_id = binding.env_init(
                self.observations[start:end],
                self.actions[start:end],
                self.rewards[start:end],
                self.terminals[start:end],
                self.truncations[start:end],
                i + seed * num_envs,
                width=width,
                height=height,
                ball_radius=ball_radius,
                pocket_radius=pocket_radius,
                friction=friction,
                restitution=restitution,
                impulse=impulse,
                min_power=min_power,
                reward_step=reward_step,
                reward_shot=reward_shot,
                reward_legal_pot=reward_legal_pot,
                reward_illegal_pot=reward_illegal_pot,
                reward_foul=reward_foul,
                reward_win=reward_win,
                fast_forward=fast_forward,
                max_physics_steps=max_physics_steps,
                max_steps=max_steps,
            )
            c_envs.append(env_id)

        self.c_envs = binding.vectorize(*c_envs)

    def reset(self, seed=None):
        self.tick = 0
        binding.vec_reset(self.c_envs, 0 if seed is None else seed)
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        self.tick += 1
        binding.vec_step(self.c_envs)

        info = []
        if self.tick % self.report_interval == 0:
            info.append(binding.vec_log(self.c_envs))

        return (
            self.observations,
            self.rewards,
            self.terminals,
            self.truncations,
            info,
        )

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)


def test_performance(timeout=10, atn_cache=1024):
    num_envs = 1024
    env = Pool(num_envs=num_envs)
    env.reset()

    tick = 0
    actions = np.empty((atn_cache, env.num_agents, 2), dtype=np.int32)
    actions[..., 0] = np.random.randint(0, 16, (atn_cache, env.num_agents), dtype=np.int32)
    actions[..., 1] = np.random.randint(0, 5, (atn_cache, env.num_agents), dtype=np.int32)

    import time

    start = time.time()
    while time.time() - start < timeout:
        env.step(actions[tick % atn_cache])
        tick += 1

    sps = env.num_agents * tick / (time.time() - start)
    print(f'SPS: {sps:,}')


if __name__ == '__main__':
    test_performance()

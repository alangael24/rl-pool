import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.pool import binding


class Pool(pufferlib.PufferEnv):
    def __init__(
        self,
        num_envs=1,
        width=2.84,
        height=1.42,
        ball_radius=0.03,
        pocket_radius=0.06,
        friction=0.992,
        restitution=0.96,
        impulse=0.12,
        reward_step=-0.001,
        reward_pot_object=1.0,
        reward_scratch=-0.5,
        max_steps=300,
        report_interval=64,
        render_mode='human',
        buf=None,
        seed=0,
    ):
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.report_interval = report_interval
        self.tick = 0

        self.single_observation_space = gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(11,),
            dtype=np.float32,
        )
        # Action 0: no-op, 1..16: shot direction around 360 degrees
        self.single_action_space = gymnasium.spaces.Discrete(17)

        super().__init__(buf)

        self.c_envs = binding.vec_init(
            self.observations,
            self.actions,
            self.rewards,
            self.terminals,
            self.truncations,
            num_envs,
            seed,
            width=width,
            height=height,
            ball_radius=ball_radius,
            pocket_radius=pocket_radius,
            friction=friction,
            restitution=restitution,
            impulse=impulse,
            reward_step=reward_step,
            reward_pot_object=reward_pot_object,
            reward_scratch=reward_scratch,
            max_steps=max_steps,
        )

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
    num_envs = 4096
    env = Pool(num_envs=num_envs)
    env.reset()

    tick = 0
    actions = np.random.randint(0, 17, (atn_cache, num_envs), dtype=np.int32)

    import time

    start = time.time()
    while time.time() - start < timeout:
        env.step(actions[tick % atn_cache])
        tick += 1

    sps = num_envs * tick / (time.time() - start)
    print(f'SPS: {sps:,}')


if __name__ == '__main__':
    test_performance()

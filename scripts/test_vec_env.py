from time import time

import torch
from tqdm import trange
from gym.vector.async_vector_env import AsyncVectorEnv

from scripts.make_envs import make_cart_pole, make_cheetah, make_crafter
from nn.actors import ActorCartPole, ActorCheetah, ActorCrafter


device = torch.device('cpu')
n_frames = 10_000


def make_vec_env(make_env, n_envs):
    return AsyncVectorEnv([make_env for _ in range(n_envs)])


def step(obs, env, actor):
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
    with torch.no_grad():
        action = actor.act(obs_t).cpu().numpy()
    obs, _, done, _ = env.step(action)
    return obs


def main():
    for skip, env_name, make_env, make_actor in zip(
        [True, True, False],
        ['CartPole', 'HalfCheetah', 'crafter'],
        [make_cart_pole, make_cheetah, make_crafter],
        [ActorCartPole, ActorCheetah, ActorCrafter]
    ):
        if not skip:
            print(f'testing_{env_name}...')
            best_time, best_n = float('inf'), None
            actor = make_actor()
            for n_envs in [2, 4, 8, 16, 32, 64, 128]:
                vec_env = make_vec_env(make_env, n_envs)
                start_time = time()
                obs = vec_env.reset()
                p_bar = trange(n_frames // n_envs)
                p_bar.set_description(f'n_envs = {n_envs}')
                for _ in p_bar:  # number of frames = number of envs * number of steps
                    obs = step(obs, vec_env, actor)
                total_time = time() - start_time
                if total_time < best_time:
                    best_time = total_time
                    best_n = n_envs
                vec_env.close()
                del vec_env
            print(f'done, best time = {best_time}, best_n = {best_n}, fps = {n_frames / best_time}')
            # CartPole: 10, 14, 22, 33, 43, 52
            # HalfCheetah: 6.8, 11, 17, 23, 29, 36
            # qbert @ 100k (2,   4,  8, 16, 32, 64, 128):
            # time:         49, 50, 46, 41, 32, 15, 7; 507 fps


if __name__ == '__main__':
    main()

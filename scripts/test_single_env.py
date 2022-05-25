from time import time

import torch
from tqdm import trange

from scripts.make_envs import make_cart_pole, make_cheetah, make_crafter
from nn.actors import ActorCartPole, ActorCheetah, ActorCrafter

device = torch.device('cpu')
n_frames = 10_000


def step(obs, env, actor):
    # add batch dim to the obs and remove it from the action, do env step
    obs_t = torch.tensor(obs[None], dtype=torch.float32, device=device)
    with torch.no_grad():
        action = actor.act(obs_t).cpu().numpy()[0]
    obs, _, done, _ = env.step(action)
    if done:
        obs = env.reset()
    return obs


def main():
    for skip, env_name, make_env, make_actor in zip(
        [True, True, False],
        ['CartPole', 'HalfCheetah', 'Crafter'],
        [make_cart_pole, make_cheetah, make_crafter],
        [ActorCartPole, ActorCheetah, ActorCrafter]
    ):
        if not skip:
            env = make_env()
            actor = make_actor()

            start_time = time()
            obs = env.reset()
            for _ in trange(n_frames):
                obs = step(obs, env, actor)

            total_time = time() - start_time
            print(f'{env_name} took {total_time}, fps = {n_frames / total_time}')
            # CartPole: 20600
            # HalfCheetah: 13000 - wow, this number is pretty huge
            # Crafter: 200 FPS


if __name__ == '__main__':
    main()

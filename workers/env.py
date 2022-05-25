from typing import Callable
from multiprocessing.connection import Connection

import gym
import numpy as np
import torch


class EnvWorker:
    def __init__(
            self,
            worker_id: int,
            make_env: Callable[[], gym.Env],
            num_envs: int,
            device: torch.device,
            connection: Connection
    ):
        """
        Worker with multiple environments which steps one after another, i.e. sequentially.

        Args:
            worker_id: special ID number for worker. May be useful during batch-forming.
            make_env: environment factory.
            num_envs: number of environments in one process.
            device: torch device to put data on.
            connection: connection with external process.
                        Accepts commands and data through it and sends results back.
        """
        self._worker_id = worker_id
        self._envs = [make_env() for _ in range(num_envs)]
        self._device = device
        self._connection = connection
        self._available_commands = ['reset', 'step', 'close']

    @property
    def available_commands(self):
        return self._available_commands

    def _reset(self):
        states = np.asarray([env.reset() for env in self._envs], dtype=np.float32)
        states = torch.tensor(states, device=self._device)
        return states

    def _step(self, actions):
        new_state, reward, done = [], [], []
        for env, action in zip(self._envs, actions):
            s, r, d, _ = env.step(action)
            if d:
                s = env.reset()
            new_state.append(s)
            reward.append(r)
            done.append(d)
        new_state = torch.tensor(np.asarray(new_state, dtype=np.float32), device=self._device)
        reward = torch.tensor(np.asarray(reward), device=self._device)
        done = torch.tensor(np.asarray(done), device=self._device)
        return new_state, reward, done

    def _close(self):
        [env.close() for env in self._envs]

    def work(self):
        try:
            while True:
                if self._connection.poll():
                    cmd, data = self._connection.recv()
                    if cmd == 'reset':
                        reset_result = self._reset()
                        self._connection.send((self._worker_id, reset_result))
                    elif cmd == 'step':
                        actions = data.cpu().numpy()
                        new_state, reward, done = self._step(actions)
                        self._connection.send((self._worker_id, new_state))
                        # self._connection.send((self._worker_id, new_state, reward, done))
                    elif cmd == 'close':
                        self._close()
                        self._connection.send((self._worker_id, True))
                        break
                    else:
                        raise NotImplementedError
        except KeyboardInterrupt:
            print(f'env worker {self._worker_id} got KeyboardInterrupt')

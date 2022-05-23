from typing import Callable

import gym
import torch.nn as nn
from torch.multiprocessing import Process, Pipe, Queue

from workers import EnvWorker, ModelWorker


class Throughput:
    def __init__(
            self,
            make_env: Callable[[], gym.Env],
            make_model: Callable[[], nn.Module],
            n_env_workers: int,
            n_model_workers: int,
            n_envs_per_process,
            out_connection: Queue
    ):
        self.env_connections, self._env_processes = [], []
        for env_worker_id in range(n_env_workers):
            connection = self._init_env_worker(
                env_worker_id, make_env, n_envs_per_process
            )
            self.env_connections.append(connection)

        self._put_to_model = Queue()
        self._get_from_model = Queue()
        # self._get_from_model = ffQ()

        self.frames_counter = out_connection

        self._model_processes = []
        for model_worker_id in range(n_model_workers):
            self._init_model_worker(
                model_worker_id, make_model, self._put_to_model, self._get_from_model
            )
        self._model_free = [True for _ in range(n_model_workers)]

    @staticmethod
    def _init_env_worker(env_worker_id, make_env, n_evs_per_worker):
        connection_1, connection_2 = Pipe()
        worker = EnvWorker(env_worker_id, make_env, n_evs_per_worker, connection_2)
        process = Process(target=worker.work)
        process.start()
        return connection_1

    @staticmethod
    def _init_model_worker(model_worker_id, make_model, put_to_model, get_from_model):
        # TODO: add server connection
        worker = ModelWorker(model_worker_id, make_model, put_to_model, get_from_model)
        process = Process(target=worker.work)
        process.start()

    def env_to_model(self):
        # continuously recv data from env workers and sends them to model workers
        try:
            while True:
                # TODO: how to stop?
                for connection in self.env_connections:
                    if connection.poll():
                        env_id, obs = connection.recv()
                        self._put_to_model.put(('act', (env_id, obs)))
                        self.frames_counter.put(obs.shape[0])
        except KeyboardInterrupt:
            print('throughput: env to model interrupted')

    def model_to_env(self):
        # continuously recv actions from models and sends them to environments
        try:
            while True:
                # TODO: how to stop?
                if not self._get_from_model.empty():
                    env_id, action = self._get_from_model.get()
                    self.env_connections[env_id].send(('step', action))
        except KeyboardInterrupt:
            print('throughput: model to env interrupted')

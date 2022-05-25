from typing import Callable

import gym
import torch
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
            n_envs_per_process: int,
            out_connection: Queue,  # send results through it
            in_connection: Queue  # recv signals through it
    ):
        self.env_connections = []
        for env_worker_id in range(n_env_workers):
            connection = self._init_env_worker(
                env_worker_id, make_env, n_envs_per_process
            )
            self.env_connections.append(connection)

        self._put_to_model = Queue()
        self._get_from_model = Queue()

        self._out_connection = out_connection
        self._in_connection = in_connection

        self._n_model_workers = n_model_workers
        self._model_server_connections = []
        self._model_processes = []
        for model_worker_id in range(n_model_workers):
            connection = self._init_model_worker(
                model_worker_id, make_model, self._put_to_model, self._get_from_model
            )
            self._model_server_connections.append(connection)
        # TODO: need to store all collected data in some buffer,
        #  I think good places to do this are model and env workers

    @staticmethod
    def _init_env_worker(env_worker_id, make_env, n_evs_per_worker):
        connection_1, connection_2 = Pipe()
        device = torch.device('cpu')
        worker = EnvWorker(env_worker_id, make_env, n_evs_per_worker, device, connection_2)
        process = Process(target=worker.work)
        process.start()
        return connection_1

    @staticmethod
    def _init_model_worker(model_worker_id, make_model, put_to_model, get_from_model):
        connection_1, connection_2 = Pipe()
        worker = ModelWorker(model_worker_id, make_model, put_to_model, get_from_model, connection_2)
        process = Process(target=worker.work)
        process.start()
        return connection_1

    def env_to_model(self):
        # continuously recv data from env workers and sends them to model workers
        for connection in self.env_connections:
            connection.send(('reset', None))

        try:
            while True:
                for connection in self.env_connections:
                    if connection.poll():
                        # TODO: after closing env connection could return True instead of obs
                        #  need to handle it somehow
                        env_id, obs = connection.recv()
                        self._put_to_model.put(('act', (env_id, obs)))
                        self._out_connection.put(obs.shape[0])
                if self.stop():
                    # TODO: how to close all model workers
                    #  if they all got the same Queue to listen to?
                    #  probably next two lines will do the job,
                    #  because one model process shouldn't be able to get
                    #  data from Queue after receiving 'close' command.
                    for _ in range(self._n_model_workers):
                        self._put_to_model.put(('close', None))
                    break
        except KeyboardInterrupt:
            print('throughput: env to model interrupted')

    def model_to_env(self):
        # continuously recv actions from models and sends them to environments
        try:
            while True:
                if not self._get_from_model.empty():
                    env_id, action = self._get_from_model.get()
                    self.env_connections[env_id].send(('step', action))

                if self.stop():
                    for connection in self.env_connections:
                        connection.send(('close', None))
                    break
        except KeyboardInterrupt:
            print('throughput: model to env interrupted')

    def stop(self):
        # need to put two 'stop' signals into the out_connection to stop both env_to_model and model_to_env processes
        if not self._in_connection.empty():
            cmd = self._in_connection.get()
            if cmd == 'stop':
                return True
        return False

from typing import Callable
from multiprocessing.connection import Connection

import torch
import torch.nn as nn
from torch.multiprocessing import Queue


class ModelWorker:
    def __init__(
            self,
            worker_id: int,
            make_model: Callable[[], nn.Module],
            get_connection: Queue,
            put_connection: Queue,
            server_connection: Connection
    ):
        """
        Args:
            worker_id: number of the worker instance.
            make_model: model factory.
            get_connection: connection to get observations from.
            put_connection: connection to sent actions to.
            server_connection: connection with the model server,
                               which periodically send updated model parameters.
        """
        self._worker_id = worker_id
        self._model_instance = make_model()
        self._model_instance.share_memory()
        self._get_connection = get_connection
        self._put_connection = put_connection
        self._server_connection = server_connection

    def _act(self, observation):
        with torch.no_grad():
            action = self._model_instance.act(observation)
        return action

    def work(self):
        try:
            while True:
                if not self._get_connection.empty():
                    cmd, data = self._get_connection.get()
                    if cmd == 'act':
                        env_id, observation = data
                        action = self._act(observation)
                        self._put_connection.put((env_id, action))
                    elif cmd == 'close':
                        self._put_connection.put((self._worker_id, True))
                        break
                    else:
                        raise NotImplementedError
                if self._server_connection.poll():
                    model_state_dict = self._server_connection.recv()
                    self._model_instance.load_state_dict(model_state_dict)
        except KeyboardInterrupt:
            print(f'model worker {self._worker_id} got KeyboardInterrupt')

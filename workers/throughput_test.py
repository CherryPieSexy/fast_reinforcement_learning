from time import time
from multiprocessing import Process, Queue

import tqdm
import torch.multiprocessing as mp

from workers import Throughput
from scripts.make_envs import make_crafter
from nn.actors import ActorCrafter


def work(throughput, frames_counter, stop_connection, n_frames):
    env_to_model_process = Process(target=throughput.env_to_model)
    model_to_env_process = Process(target=throughput.model_to_env)
    env_to_model_process.start()
    model_to_env_process.start()

    start_time = time()

    frames_collected = 0
    p_bar = tqdm.tqdm(desc=f'frames_collected = {frames_collected}, fps = {frames_collected / 1}')
    try:
        while True:
            while not frames_counter.empty():
                frames_collected += frames_counter.get()
            p_bar.set_description(
                desc=f'frames_collected = {frames_collected},'
                     f'fps = {frames_collected / (time() - start_time)}'
            )
            if frames_collected >= n_frames:
                elapsed_time = time() - start_time
                print(
                    f'done, {n_frames} frames collected in {elapsed_time}, '
                    f'fps = {n_frames / elapsed_time}'
                )

                # put two stops - one for model worker and one for env worker
                stop_connection.put('stop')
                stop_connection.put('stop')
                break

    except KeyboardInterrupt:
        print('throughput interrupted')

    env_to_model_process.join()
    model_to_env_process.join()


def main():
    n_env_total = 64
    n_env_workers = 8
    n_env_per_worker = n_env_total // n_env_workers
    n_model_workers = 1
    n_frames = 100_000

    frames_counter, stop_connection = Queue(), Queue()
    throughput = Throughput(
        make_crafter, ActorCrafter,
        n_env_workers, n_model_workers, n_env_per_worker,
        frames_counter, stop_connection
    )
    work(throughput, frames_counter, stop_connection, n_frames)


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()

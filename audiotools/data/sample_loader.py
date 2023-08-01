from pathlib import Path
from typing import Callable, Tuple
from typing import Dict
from typing import List
from typing import Union
from random import shuffle
from fs.osfs import OSFS
from fs.mountfs import MountFS
from fs import open_fs
import numpy as np
from torch.utils.data import SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from ..core import AudioSignal
from ..core import util
import ray


def create_file_system(sources: list[Tuple[str, Union[str, Path]]]) -> MountFS:
    dataset_fs = MountFS()
    [dataset_fs.add_fs(source[0], open_fs(source[1])) for source in sources]
    return dataset_fs


# Get all music files and shuffle them
music_files = []
for path in dataset_fs.walk.files(filter=["*.mp3"]):
    music_files.append(path)
    # print(path)
random.shuffle(music_files)
print(music_files[:50])
pass


# Background worker/actor to generate samples
@ray.remote
class SampleLoaderWorker:
    def __init__(
        self,
        sample_rate: int = 16000,
        duration: float = None,
        loudness_cutoff: float = -40,
        state: str = "full",
        transform: Callable = None,
        num_channels: int = 1,
        matcher: Callable = None,
        offset: float = None,
        sampler: str = "sequential",
        shuffle: bool = False,
        sources: List[Tuple[str, Union[str, Path]]] = [],
        seed: int = 0,
    ):
        self.sample_rate = sample_rate
        self.duration = duration
        self.loudness_cutoff = loudness_cutoff
        self.state = state
        self.transform = transform
        self.matcher = matcher
        self.offset = offset
        self.sampler = sampler
        self.shuffle = shuffle
        self.seed = seed
        self.num_channels = num_channels
        self.sources: List[Tuple[str, str | Path]] = sources

        # Each worker needs its represntation of the file system due to potential remoting.
        self.dataset_fs = create_file_system(sources=self.sources)

    def load_sample(self, path):
        signal = AudioSignal.zeros(self.duration, self.sample_rate, self.num_channels)
        if path != None and path != "none":
            with self.dataset_fs.open(path, "rb") as file:
                if self.offset is None:
                    signal = AudioSignal.salient_excerpt(
                        file,
                        duration=self.duration,
                        state=self.state,
                        loudness_cutoff=self.loudness_cutoff,
                    )
                else:
                    signal = AudioSignal(
                        file,
                        offset=self.offset,
                        duration=self.duration,
                    )

        return signal


class SampleLoader:
    def __init__(
        self,
        sources: List[Union[str, Path]],
        sample_rate: int = 16000,
        duration: float = None,
        loudness_cutoff: float = -40,
        state: str = "full",
        transform: Callable = None,
        num_channels: int = 1,
        matcher: Callable = None,
        offset: float = None,
        sampler: str = "sequential",
        shuffle: bool = False,
        seed: int = 0,
        worker_num: int = 16,
    ):
        self.sources = [Path(s) for s in sources]
        self.sample_rate = sample_rate
        self.duration = duration
        self.loudness_cutoff = loudness_cutoff
        self.state = state
        self.transform = transform
        self.matcher = matcher
        self.offset = offset
        self.sampler = sampler
        self.shuffle = shuffle
        self.seed = seed
        self.num_channels = num_channels

        self.worker_num = worker_num
        self.workers = [
            SampleLoaderWorker.remote(
                path="test.wav",
                state=self.state,
                duration=self.duration,
                loudness_cutoff=self.loudness_cutoff,
                offset=self.offset,
                sample_rate=self.sample_rate,
                num_channels=self.num_channels,
            )
            for _ in range(self.worker_num)
        ]
        self.actor_pool = ray.util.ActorPool(self.workers)
        self.pending_list = []

    def start(self):
        # Start generating samples
        self.pending_list.extend(worker.load_sample() for worker in self.workers)

    def get_samples(self, paths: List[str, Path]):
        # Get samples from the background worker
        ready_refs, self.pending_list = ray.wait(
            self.pending_list, 
            num_returns=num_samples
        )

        self.actor_pool.map(lambda a, v: a.load_sample.remote(v), [1, 2, 3, 4])
        self.pending_list.extend(worker.load_sample() for worker in self.workers)

        results = ray.get(ready_refs)

        return ready_refs

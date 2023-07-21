from pathlib import Path
from typing import Callable
from typing import Dict
from typing import List
from typing import Union

import numpy as np
from torch.utils.data import SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from ..core import AudioSignal
from ..core import util

class AudioZipLoader:
    """Loads audio endlessly from a list of audio sources
    stored in Zip files Follows same general behavior as regular loader.

    Parameters
    ----------
    sources : List[str], optional
        Sources containing folders, or CSVs with
        paths to audio files, by default None
    weights : List[float], optional
        Weights to sample audio files from each source, by default None
    relative_path : str, optional
        Path audio should be loaded relative to, by default ""
    transform : Callable, optional
        Transform to instantiate alongside audio sample,
        by default None
    ext : List[str]
        List of extensions to find audio within each source by. Can
        also be a file name (e.g. "vocals.wav"). by default
        ``['.wav', '.flac', '.mp3', '.mp4']``.
    shuffle: bool
        Whether to shuffle the files within the dataloader. Defaults to True.
    shuffle_state: int
        State to use to seed the shuffle of the files.
    """

    def __init__(
        self,
        sources: List[str] = None,
        weights: List[float] = None,
        transform: Callable = None,
        relative_path: str = "",
        ext: List[str] = util.AUDIO_EXTENSIONS,
        shuffle: bool = True,
        shuffle_state: int = 0,
    ):
        self.audio_lists = util.read_sources(
            sources, relative_path=relative_path, ext=ext
        )

        self.audio_indices = [
            (src_idx, item_idx)
            for src_idx, src in enumerate(self.audio_lists)
            for item_idx in range(len(src))
        ]
        if shuffle:
            state = util.random_state(shuffle_state)
            state.shuffle(self.audio_indices)

        self.sources = sources
        self.weights = weights
        self.transform = transform

    def __call__(
        self,
        state,
        sample_rate: int,
        duration: float,
        loudness_cutoff: float = -40,
        num_channels: int = 1,
        offset: float = None,
        source_idx: int = None,
        item_idx: int = None,
        global_idx: int = None,
    ):
        if source_idx is not None and item_idx is not None:
            try:
                audio_info = self.audio_lists[source_idx][item_idx]
            except:
                audio_info = {"path": "none"}
        elif global_idx is not None:
            source_idx, item_idx = self.audio_indices[
                global_idx % len(self.audio_indices)
            ]
            audio_info = self.audio_lists[source_idx][item_idx]
        else:
            audio_info, source_idx, item_idx = util.choose_from_list_of_lists(
                state, self.audio_lists, p=self.weights
            )

        path = audio_info["path"]
        signal = AudioSignal.zeros(duration, sample_rate, num_channels)

        if path != "none":
            if offset is None:
                signal = AudioSignal.salient_excerpt(
                    path,
                    duration=duration,
                    state=state,
                    loudness_cutoff=loudness_cutoff,
                )
            else:
                signal = AudioSignal(
                    path,
                    offset=offset,
                    duration=duration,
                )

        if num_channels == 1:
            signal = signal.to_mono()
        signal = signal.resample(sample_rate)

        if signal.duration < duration:
            signal = signal.zero_pad_to(int(duration * sample_rate))

        for k, v in audio_info.items():
            signal.metadata[k] = v

        item = {
            "signal": signal,
            "source_idx": source_idx,
            "item_idx": item_idx,
            "source": str(self.sources[source_idx]),
            "path": str(path),
        }
        if self.transform is not None:
            item["transform_args"] = self.transform.instantiate(state, signal=signal)
        return item

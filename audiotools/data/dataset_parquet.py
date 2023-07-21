import pandas as pd
from typing import List, Callable, Dict, Union
from pathlib import Path
from pyarrow import parquet as pq
from pathlib import Path

import numpy as np
from torch.utils.data import SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from ..core import AudioSignal
from ..core import util


def read_parquet_sources(
    sources: List[str],
    remove_empty: bool = True,
    relative_path: str = "",
):
    """Reads audio sources that are parquet files
    that contain paths to audio files.
    """
    files = []
    relative_path = Path(relative_path)
    for source in sources:
        source = str(source)
        _files = []
        if source.endswith(".parquet"):
            df = pd.read_parquet(source)
            _files.extend({"path":f'{source}::{_}', "df":df, "idx":_} for _ in df.index)
        files.append(sorted(_files, key=lambda x: x["path"]))
    return files

class ParquetAudioLoader(AudioLoader):
    """Loads audio endlessly from a list of audio sources
    containing paths to audio files in parquet files.
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
        super().__init__(sources, weights, transform, relative_path, ext, shuffle, shuffle_state, read_parquet_sources )
        
        
    def load_sample(self, state, duration, loudness_cutoff, offset, path, sample_rate, num_channels):
        signal = AudioSignal.zeros(duration, sample_rate, num_channels)
        if path != "none":
            # Load audio from dataframe
            audio_data: np.ndarray = path["df"][path["idx"]]['audio']
                signal = AudioSignal.from_fileobj(f, sample_rate=sample_rate, num_channels=num_channels)
            if offset is None:
                signal = AudioSignal.salient_excerpt(
                    audio_data,
                    duration=duration,
                    state=state,
                    loudness_cutoff=loudness_cutoff,
                )
            else:
                signal = AudioSignal(
                    audio_data,
                    offset=offset,
                    duration=duration,
                )
                
        return signal
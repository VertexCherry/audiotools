import pyarrow.parquet as pq
from typing import List, Callable
from audiotools.data import util, AudioSignal

class ArrowAudioLoader(AudioLoader):
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
        # change read_sources to read_arrow_sources
        self.audio_lists = self.read_arrow_sources(
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
    
    @staticmethod
    def read_arrow_sources(sources: List[str], relative_path: str = "", ext: List[str] = util.AUDIO_EXTENSIONS):
        files = []
        relative_path = Path(relative_path)
        for source in sources:
            source = str(source)
            _files = []
            if source.endswith(".arrow"):
                table = pq.read_table(source)
                for i in range(table.num_rows):
                    x = table.slice(i, 1).to_pandas().iloc[0].to_dict()
                    if x["path"] != "":
                        x["path"] = str(relative_path / x["path"])
                    _files.append(x)
            files.append(sorted(_files, key=lambda x: x["path"]))
        return files

from datasets import load_dataset
from pathlib import Path

class HuggingAudioLoader(AudioLoader):
    def __init__(
        self,
        dataset: str,
        weights: List[float] = None,
        transform: Callable = None,
        relative_path: str = "",
        ext: List[str] = util.AUDIO_EXTENSIONS,
        shuffle: bool = True,
        shuffle_state: int = 0,
    ):
        self.dataset = load_dataset(dataset)
        
        # Convert dataset paths to expected format
        self.audio_lists = []
        for item in self.dataset:
            path = str(Path(relative_path) / item['path'])
            self.audio_lists.append({"path": path})

        self.audio_indices = [(src_idx, item_idx) for src_idx, src in enumerate(self.audio_lists) for item_idx in range(len(src))]

        if shuffle:
            state = util.random_state(shuffle_state)
            state.shuffle(self.audio_indices)

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
            source_idx, item_idx = self.audio_indices[global_idx % len(self.audio_indices)]
            audio_info = self.audio_lists[source_idx][item_idx]
        else:
            audio_info, source_idx, item_idx = util.choose_from_list_of_lists(state, self.audio_lists, p=self.weights)

        # The rest of the code remains the same as the original __call__ method.

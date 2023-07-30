from pathlib import Path
import typing
import os
import numpy as np
import soundfile as sf
import audioread

import torch

Str_Path = typing.Union[str, Path, os.PathLike[typing.Any]]
Audio_File_Type = typing.Union[sf.SoundFile, audioread.AudioFile, typing.BinaryIO]
Audio_Array_Type = typing.Union[torch.Tensor, np.ndarray]

Str_Path_Filelike = typing.Union[Str_Path, Audio_File_Type]
Str_Path_Array = typing.Union[Str_Path, Audio_Array_Type]
All_Audio_Types = typing.Union[Str_Path_Filelike, Audio_Array_Type]


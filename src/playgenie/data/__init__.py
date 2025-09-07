from typing import *
from dataclasses import dataclass,field
import numpy as np


@dataclass
class Songs:
    names :List[int]
    embeddings:Dict[int, np.ndarray]

    def __len__(self) -> int:
        return len(self.names)

@dataclass
class Playlists:
    names: List[int]
    songs: Dict[int, List[Songs]]

    def __len__(self) -> int:
        return len(self.names)

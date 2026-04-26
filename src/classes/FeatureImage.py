import cv2
import os
import numpy as np
from abc import ABC, abstractmethod
from typing import TypeAlias

from core.types import KPGroup, KnnMatchesList


FeatureDescriptors: TypeAlias = np.ndarray


class FeatureImage(ABC):

    def __init__(self, file_path: str) -> None:
        self._file_path = file_path
        self._shape: tuple[int, int]
        self._kps: KPGroup | None = None
        self._descriptors: FeatureDescriptors | None = None
        self._matches: KnnMatchesList | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape
    
    @property
    def kps(self) -> KPGroup | None:
        return self._kps

    @property
    def descriptors(self) -> FeatureDescriptors | None:
        return self._descriptors

    @property
    def n_kps(self) -> int:
        if self._kps is not None:
            return len(self._kps)
        return 0

    @property
    def filename(self) -> str:
        return os.path.basename(self._file_path)

    @property
    def matches(self) -> KnnMatchesList | None:
        return self._matches

    @abstractmethod
    def detect(self) -> None:
        pass

    def set_matches(self, matches: KnnMatchesList) -> None:
        if len(matches) != self.n_kps:
            raise ValueError("Length of matches does not match length of self._kps")
        self._matches = matches
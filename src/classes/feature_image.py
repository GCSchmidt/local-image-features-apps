import cv2
import os
import numpy as np
from abc import ABC, abstractmethod
from typing import TypeAlias

from core.types import KPGroup, KnnMatchesList


FeatureDescriptors: TypeAlias = np.ndarray


class FeatureImage(ABC):

    def __init__(self, file_path: str, max_dim: int | None = None) -> None:
        self._file_path = file_path
        self._max_dim = max_dim
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

    def _resize_image(self, img: np.ndarray) -> np.ndarray:
        if self._max_dim is None:
            return img
        h, w = img.shape[:2]
        longest = max(h, w)
        if longest <= self._max_dim:
            return img
        scale = self._max_dim / longest
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    @abstractmethod
    def detect(self) -> None:
        pass

    def set_matches(self, matches: KnnMatchesList) -> None:
        if len(matches) != self.n_kps:
            raise ValueError("Length of matches does not match length of self._kps")
        self._matches = matches

    def get_kp_range_from_matches(self)-> tuple[int, int]:
        if self._matches is not None:
            start_i = self._matches[0][0].queryIdx
            end_id = self._matches[-1][0].queryIdx
            return (start_i, end_id)
        return (0, 0)
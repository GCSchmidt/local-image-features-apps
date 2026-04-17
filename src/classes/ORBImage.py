"""

"""

import cv2
import numpy as np

from core.types import ORBDescriptors, KPGroup


class ORBImage():

    __orb = cv2.ORB_create()

    def __init__(self, file_path: str) -> None:
        self._file_path = file_path
        self._kps: KPGroup | None = None
        self._descriptors: ORBDescriptors | None = None

    @property
    def kps(self):
        return self._kps

    @property
    def descriptors(self):
        return self._descriptors

    def detect(self):
        img = cv2.imread(self._file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {self._file_path}")
        kps, desc = ORBImage.__orb.detectAndCompute(img, None)
        if kps:
            self._kps = kps
        if desc is not None:
            self._descriptors = desc

    @property
    def n_kps(self) -> int:
        if self._kps is not None:
            return len(self._kps)
        else:
            return 0

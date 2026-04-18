"""

"""
import cv2
import os

from core.types import ORBDescriptors, KPGroup, KnnMatchesList


class ORBImage():

    __orb = cv2.ORB_create()

    def __init__(self, file_path: str) -> None:
        self._file_path = file_path
        self._kps: KPGroup | None = None
        self._descriptors: ORBDescriptors | None = None
        self._matches: KnnMatchesList | None = None

    @property
    def kps(self):
        return self._kps

    @property
    def descriptors(self):
        return self._descriptors
    
    @property
    def matches(self):
        return self._matches

    @property
    def n_kps(self) -> int:
        if self._kps is not None:
            return len(self._kps)
        else:
            return 0

    @property
    def filename(self) -> str:
        return os.path.basename(self._file_path)

    def detect(self) -> None:
        img = cv2.imread(self._file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {self._file_path}")
        kps, desc = ORBImage.__orb.detectAndCompute(img, None)
        if kps:
            self._kps = kps
        if desc is not None:
            self._descriptors = desc

    def set_matches(self, matches: KnnMatchesList) -> None:
        if len(matches) != self.n_kps:
            raise Exception("Length of matches does not match length of self._kps")
        else:
            self._matches = matches
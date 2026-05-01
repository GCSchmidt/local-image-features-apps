"""

"""
import cv2

from classes.FeatureImage import FeatureImage


class SiftImage(FeatureImage):

    __sift = cv2.SIFT_create()

    def __init__(self, file_path: str, max_dim: int | None = None) -> None:
        super().__init__(file_path, max_dim)

    def detect(self) -> None:
        img = cv2.imread(self._file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {self._file_path}")
        img = self._resize_image(img)
        self._shape = img.shape[:2]
        kps, desc = SiftImage.__sift.detectAndCompute(img, None)
        if kps:
            self._kps = kps
        if desc is not None:
            self._descriptors = desc
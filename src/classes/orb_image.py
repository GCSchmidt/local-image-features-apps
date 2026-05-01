"""

"""
import cv2

from classes.feature_image import FeatureImage


class ORBImage(FeatureImage):

    __orb = cv2.ORB_create(
        nfeatures=1000,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=31,
        fastThreshold=20,
        scoreType=cv2.ORB_HARRIS_SCORE
    )

    def __init__(self, file_path: str, max_dim: int | None = None) -> None:
        super().__init__(file_path, max_dim)

    def detect(self) -> None:
        img = cv2.imread(self._file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {self._file_path}")
        img = self._resize_image(img)
        self._shape = img.shape[:2]
        kps, desc = ORBImage.__orb.detectAndCompute(img, None)
        if kps:
            self._kps = kps
        if desc is not None:
            self._descriptors = desc
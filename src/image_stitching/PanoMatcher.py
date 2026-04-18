import os
import cv2
import numpy as np
from core.constants import KNN_FOR_PANORAMA, M_CANDIDATE_IMAGES
import image_stitching.pipeline as pipeline
from classes.ORBImage import ORBImage


class PanoMatcher:

    def __init__(self, image_paths: list[str]) -> None:
        self._orb_images = [ORBImage(path) for path in image_paths]

    def detect_features(self):
        for oi in self._orb_images:
            oi.detect()

    def generate_pano(self):
        self.detect_features()
        match_counts = self.match_images()

    def match_images(self) -> np.ndarray:
        matches = pipeline.match_orb_images(self._orb_images, KNN_FOR_PANORAMA*2)
        processed_matches, match_counts = pipeline.process_matches(self.orb_images, matches)
        pipeline.assign_matches_to_orb_images(self._orb_images, processed_matches)
        return match_counts

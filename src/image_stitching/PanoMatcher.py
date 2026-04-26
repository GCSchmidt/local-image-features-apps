import os
import logging
import cv2
import numpy as np
from core.constants import KNN_FOR_PANORAMA, M_CANDIDATE_IMAGES, INLIER_THRESHOLD, MIN_INLIERS
from core.enums import FeatureDetectorType
import image_stitching.pipeline as pipeline
from image_stitching.PanoGraph import ImageConnection, PanoGraph
from classes.ORBImage import ORBImage
from classes.SiftImage import SiftImage
from classes.FeatureImage import FeatureImage
from utils import file_utils



logger = logging.getLogger("image_stitching")


class PanoMatcher:

    def __init__(self, image_paths: list[str], detector: type[FeatureImage] = SiftImage) -> None:
        self._images = [detector(path) for path in image_paths]
        self._detector = detector
        log_msg_lines = ["ImgIds : Image File Names"]
        for i in range(len(self._images)):
            log_msg_lines.append(f"{i} : {self._get_image_name(i)}")
        logger.debug(msg="\n".join(log_msg_lines))

    def _get_image_name(self, id):
        return self._images[id].filename

    def _detect_features(self):
        for img in self._images:
            img.detect()

    def generate_panorama(self):
        self._detect_features()
        match_counts = self._match_images()
        logger.debug(f"match count:\n{str(match_counts)}")
        self._establish_connections(match_counts)

    def _match_images(self) -> np.ndarray:
        matches = pipeline.match_images(self._images, KNN_FOR_PANORAMA*2)
        processed_matches, match_counts = pipeline.process_matches(self._images, matches)
        pipeline.assign_matches_to_images(self._images, processed_matches)
        return match_counts

    def _establish_connections(self, match_counts: np.ndarray):
        n_images = len(self._images)
        inlier_count = np.zeros(match_counts.shape)
        overlap_count = np.zeros(match_counts.shape)
        for img_id1 in range(n_images):
            best_matched_images = pipeline.get_best_image_matches(match_counts, img_id1)
            log_msg = f"best matches with {self._get_image_name(img_id1)}: "
            log_msg += " ".join([self._get_image_name(i) for i in best_matched_images])
            logger.debug(log_msg)
            
            for img_id2 in best_matched_images:
                H, n_inliers, n_overlapping = self._get_homography(img_id1, img_id2)
                log_msg = f"Match bewteen {self._get_image_name(img_id1)} and {self._get_image_name(img_id2)}:"
                log_msg += f"\nH:{str(H)}\ninliers: {n_inliers}, \noverlapping: {n_overlapping}"
                logger.debug(log_msg)
                inlier_count[img_id1][img_id2] = n_inliers
                overlap_count[img_id1][img_id2] = n_overlapping

        valid = overlap_count > MIN_INLIERS
        valid = inlier_count > MIN_INLIERS
        connections = np.zeros_like(inlier_count, dtype=bool)
        connections[valid] = inlier_count[valid] > INLIER_THRESHOLD * overlap_count[valid]

        logger.debug(msg=f"inlier count:\n{str(inlier_count)}")
        logger.debug(msg=f"overlap count:\n{str(overlap_count)}")
        logger.debug(msg=f"connected:\n{str(connections)}")

        log_lines = []
        for r, row in enumerate(connections):
            connected_images = np.where(row)[0]
            line = f"{self._get_image_name(r)} is connected to: "
            line += " ".join([self._get_image_name(id) for id in connected_images])
            log_lines.append(line)
        log_msg = "\n".join(log_lines)
        logger.debug(msg=f"Image Connections:\n{log_msg}")

    def _get_homography(self, img_id1: int, img_id2: int) -> tuple[np.ndarray, int, int]:
        img1 = self._images[img_id1]
        img2 = self._images[img_id2]

        if img1.matches is None or img1.kps is None or img2.kps is None:
            return np.identity(3), 0, 0

        kp_ranges = pipeline.get_kps_ranges(self._images)

        matched_pairs: list[tuple[int, int]] = []
        for kp_id1 in range(img1.n_kps):
            matchlist = pipeline.keep_relevant_matches(img1.matches[kp_id1], img_id2, kp_ranges)
            if (matchlist) and pipeline.is_good_match(matchlist):
                train_kp_id = matchlist[0].trainIdx
                matched_kp_id2 = train_kp_id - kp_ranges[img_id2]
                matched_pairs.append((kp_id1, matched_kp_id2))

        log_msg = "Number of Good Matches between"
        log_msg += f" {self._get_image_name(img_id1)} and {self._get_image_name(img_id2)}:"
        log_msg += f" {len(matched_pairs)}"
        logger.debug(log_msg)
        
        if len(matched_pairs) < 4:
            return np.identity(3), 0, 0

        src = np.empty((len(matched_pairs), 2), dtype=np.float32)
        dst = np.empty((len(matched_pairs), 2), dtype=np.float32)

        kps1 = img1.kps
        kps2 = img2.kps

        for i, (kp_id1, kp_id2) in enumerate(matched_pairs):
            src[i, 0] = kps1[kp_id1].pt[0]
            src[i, 1] = kps1[kp_id1].pt[1]
            dst[i, 0] = kps2[kp_id2].pt[0]
            dst[i, 1] = kps2[kp_id2].pt[1]

        H, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, 5.0, maxIters=500, confidence=0.999)

        if H is None:
            return np.identity(3), 0, 0

        n_inliers = np.count_nonzero(mask)

        n_overlapping = pipeline.count_features_in_overlap(img1, img2, H)
        
        return H, n_inliers, n_overlapping
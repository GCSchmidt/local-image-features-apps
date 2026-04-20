import os
import logging
import cv2
import numpy as np
from core.constants import KNN_FOR_PANORAMA, M_CANDIDATE_IMAGES
import image_stitching.pipeline as pipeline
from classes.ORBImage import ORBImage
from utils import file_utils


# Create a logger for this module
logger = logging.getLogger("image_stitching")


class PanoMatcher:

    def __init__(self, image_paths: list[str]) -> None:
        self._orb_images = [ORBImage(path) for path in image_paths]
        log_msg_lines = ["ImgIds : Image File Names"]
        for i in range(len(self._orb_images)):
            log_msg_lines.append(f"{i} : {self._get_image_name(i)}")
        logger.debug(msg="\n".join(log_msg_lines))

    def _get_image_name(self, id):
        return self._orb_images[id].filename

    def _detect_features(self):
        for oi in self._orb_images:
            oi.detect()

    def generate_panorama(self):
        self._detect_features()
        match_counts = self._match_images()
        logger.debug(f"match count:\n{str(match_counts)}")
        self._establish_connections(match_counts)

    def _match_images(self) -> np.ndarray:
        matches = pipeline.match_orb_images(self._orb_images, KNN_FOR_PANORAMA*2)
        processed_matches, match_counts = pipeline.process_matches(self._orb_images, matches)
        pipeline.assign_matches_to_orb_images(self._orb_images, processed_matches)
        return match_counts

    def _establish_connections(self, match_counts: np.ndarray):
        n_images = len(self._orb_images)
        inlier_count = np.zeros(match_counts.shape)
        overlap_count = np.zeros(match_counts.shape)
        for img_id1 in range(n_images):
            best_matched_images = pipeline.get_best_image_matches(match_counts, img_id1)
            log_msg = f"best matches with {self._get_image_name(img_id1)}: "
            log_msg += " ".join([self._get_image_name(i) for i in best_matched_images])
            logger.debug(log_msg)
            
            for img_id2 in best_matched_images:
                H, n_inliers, n_overlapping = self._get_homography(img_id1, img_id2)
                inlier_count[img_id1][img_id2] = n_inliers
                overlap_count[img_id1][img_id2] = n_overlapping

        connections = inlier_count > 0.3 * overlap_count

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
        """
        Gets the Homography and number of inliers between 2 ORBImages.

        Args:
            img_id1 (int): id of source ORBImage
            img_id2 (int): id of destination ORBImage

        Returns:
            tuple[np.ndarray, int]: Homography matrix and number of inliers
        """
        orb_img1 = self._orb_images[img_id1]
        orb_img2 = self._orb_images[img_id2]

        if orb_img1.matches is None or orb_img1.kps is None or orb_img2.kps is None:
            return np.identity(3), 0

        kp_ranges = pipeline.get_kps_ranges(self._orb_images)

        matched_pairs: list[tuple[int, int]] = []
        for kp_id1 in range(orb_img1.n_kps):
            for dmatch in orb_img1.matches[kp_id1]:
                train_kp_id = dmatch.trainIdx
                img_id_of_match = pipeline.get_img_id_from_kp_id(train_kp_id, kp_ranges)
                if img_id_of_match == img_id2:
                    matched_kp_id2 = train_kp_id - kp_ranges[img_id2]
                    matched_pairs.append((kp_id1, matched_kp_id2))

        if len(matched_pairs) < 4:
            return np.identity(3), 0

        src = np.empty((len(matched_pairs), 2), dtype=np.float32)
        dst = np.empty((len(matched_pairs), 2), dtype=np.float32)

        kps1 = orb_img1.kps
        kps2 = orb_img2.kps

        for i, (kp_id1, kp_id2) in enumerate(matched_pairs):
            src[i, 0] = kps1[kp_id1].pt[0]
            src[i, 1] = kps1[kp_id1].pt[1]
            dst[i, 0] = kps2[kp_id2].pt[0]
            dst[i, 1] = kps2[kp_id2].pt[1]

        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

        if H is None:
            return np.identity(3), 0

        n_inliers = np.count_nonzero(mask)
        n_overlapping = pipeline.count_features_in_overlap(orb_img1, orb_img2, H)

        return H, n_inliers, n_overlapping
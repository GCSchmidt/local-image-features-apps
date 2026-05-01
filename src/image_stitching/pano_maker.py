import os
import logging
import cv2
import numpy as np
import networkx as nx
from core.constants import KNN_FOR_PANORAMA, M_CANDIDATE_IMAGES, INLIER_THRESHOLD, MIN_INLIERS, MAX_IMAGE_DIM
import image_stitching.pipeline as pipeline
from image_stitching.panograph import ImageConnection, PanoGraph
from image_stitching.bundle_adjuster import BundleAdjuster, BundleResult
from classes.orb_image import ORBImage
from classes.sift_image import SiftImage
from classes.feature_image import FeatureImage
from utils import file_utils


logger = logging.getLogger("image_stitching")


class PanoMaker:

    def __init__(self, image_paths: list[str], detector: type[FeatureImage] = SiftImage) -> None:
        self._images = [detector(path, max_dim=MAX_IMAGE_DIM) for path in image_paths]
        self._detector = detector
        self._pano_graph = PanoGraph()
        self._bundle_result: BundleResult | None = None
        log_msg_lines = ["Image File Names (ImgIds)"]
        for i in range(len(self._images)):
            log_msg_lines.append(f"{self._get_image_name(i)} ({i})")
        logger.debug(msg="\n".join(log_msg_lines))

    def _get_image_name(self, id):
        return self._images[id].filename

    def _detect_features(self):
        for img in self._images:
            img.detect()

    def generate_panorama(self):
        self._detect_features()
        
        kp_ranges = pipeline.get_kps_ranges(self._images)
        logger.debug(f"Feature ranges:\n{str(kp_ranges)}")

        log_msg_lines = ["Image File : Number of features"]
        for i in range(len(self._images)):
            img_name = self._get_image_name(i)
            n_features = self._images[i].n_kps
            line = f"{img_name} ({i}): {n_features}"
            log_msg_lines.append(line)
        logger.debug(msg="\n".join(log_msg_lines))

        match_counts = self._match_images()
        logger.debug(f"match count:\n{str(match_counts)}")
        self._establish_connections(match_counts)
        self._pano_graph.post_process()

        ba = BundleAdjuster(self._pano_graph, self._images)
        self._bundle_result = ba.optimize()

        img_paths = [img._file_path for img in self._images]
        pipeline.render_panorama(img_paths, self._bundle_result.homographies)

    def _match_images(self) -> np.ndarray:
        matches = pipeline.match_images(self._images, KNN_FOR_PANORAMA*2)
        processed_matches, match_counts = pipeline.process_matches(self._images, matches)
        pipeline.assign_matches_to_images(self._images, processed_matches)
        
        log_msg_lines = ["Image File : Feature Ranges"]
        for i, image in enumerate(self._images):
            img_name = self._get_image_name(i)
            ranges = image.get_kp_range_from_matches()
            line = f"{img_name} ({i}): {ranges}"
            log_msg_lines.append(line)
        logger.debug(msg="\n".join(log_msg_lines))

        return match_counts

    def _establish_connections(self, match_counts: np.ndarray):
        n_images = len(self._images)
        inlier_count = np.zeros(match_counts.shape)
        overlap_count = np.zeros(match_counts.shape)
        connections = np.zeros_like(inlier_count, dtype=bool)
        for img_id1 in range(n_images):
            best_matched_images = pipeline.get_best_image_matches(match_counts, img_id1)
            log_msg = f"best matches with {self._get_image_name(img_id1)} ({img_id1}): "
            log_msg += " ".join([f"{self._get_image_name(i)} ({i})" for i in best_matched_images])
            logger.debug(log_msg)
            
            for img_id2 in best_matched_images:
                IC = self._get_connection(img_id1, img_id2)
                log_msg = f"Match bewteen {self._get_image_name(img_id1)} ({img_id1}) and {self._get_image_name(img_id2)} ({img_id2}) :"
                log_msg += f"\nH:{str(IC.homography)}\ninliers: {IC.n_inliers}, \noverlapping: {IC.n_overlap}"
                logger.debug(log_msg)
                inlier_count[img_id1, img_id2] = IC.n_inliers
                overlap_count[img_id1, img_id2] = IC.n_overlap
                if pipeline.verify_image_connection(IC):
                    self._pano_graph.add_connection(IC)
                    connections[img_id1, img_id2] = True

        logger.debug(msg=f"inlier count:\n{str(inlier_count)}")
        logger.debug(msg=f"overlap count:\n{str(overlap_count)}")
        logger.debug(msg=f"connected:\n{str(connections)}")

        log_lines = []
        for r, row in enumerate(connections):
            connected_images = np.where(row)[0]
            line = f"{self._get_image_name(r)} ({r}) is connected to: "
            line += " ".join([f"{self._get_image_name(id)} ({id})" for id in connected_images])
            log_lines.append(line)
        log_msg = "\n".join(log_lines)
        logger.debug(msg=f"Image Connections:\n{log_msg}")
        
        
    def _get_connection(self, img_id1: int, img_id2: int) -> ImageConnection:
        img1 = self._images[img_id1]
        img2 = self._images[img_id2]

        IC = ImageConnection()
        IC.reference = img_id1
        IC.target = img_id2
        if img1.matches is None or img1.kps is None or img2.kps is None:
            return IC

        matched_pairs = pipeline.get_good_matches(img1, img2)

        log_msg = "Number of Good Matches between"
        log_msg += f" {self._get_image_name(img_id1)} and {self._get_image_name(img_id2)}:"
        log_msg += f" {len(matched_pairs)}"
        logger.debug(log_msg)
        
        if len(matched_pairs) < MIN_INLIERS:
            return IC

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
            return IC

        IC.homography = H
        IC.n_inliers = np.count_nonzero(mask)
        IC.n_overlap = pipeline.count_features_in_overlap(img1, img2, H)
        IC.inlier_mask = mask.flatten().astype(bool)

        return IC
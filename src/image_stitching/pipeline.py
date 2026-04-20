import cv2
import numpy as np
from typing import TypeAlias

import core.constants as const
from core.types import KnnMatches, KnnMatchesList, ORBDescriptors
from core.enums import MatchCountStrategy as MCS
from classes.ORBImage import ORBImage

########################
# Types
########################
ORBImageList: TypeAlias = list[ORBImage]


########################
# Matching Images
########################
def match_orb_images(orb_images: ORBImageList, k: int) -> KnnMatches:
    descriptors = combine_descriptors(orb_images)
    matches = match_features(descriptors, descriptors, k)
    return matches


def combine_descriptors(orb_images: ORBImageList) -> ORBDescriptors:
    combined_list = []

    for orb_image in orb_images:
        descriptors = orb_image.descriptors

        if descriptors is not None and len(descriptors) > 0:
            combined_list.append(descriptors)

    if not combined_list:
        return np.empty((0, 32), dtype=np.uint8)

    return np.vstack(combined_list)


def match_features(descriptors1: np.ndarray, descriptors2: np.ndarray, k: int) -> KnnMatches:
    # Define FLANN parameters and match descriptors
    # from https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
    FLANN_INDEX_LSH = 6
    index_params= dict(algorithm = FLANN_INDEX_LSH,
                    table_number = 6, # 12
                    key_size = 12,     # 20
                    multi_probe_level = 1) #2
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k)

    return matches


def process_matches(orb_images: ORBImageList, all_matches: KnnMatches):
    processed_matches = []
    
    n_images = len(orb_images)
    
    match_count = np.zeros((n_images, n_images))
    
    kp_ranges = get_kps_ranges(orb_images)

    for t_matches in all_matches:
        # t_matches: tuple of cv2.DMatch
        l_matches = []
        for mid, omatch in enumerate(t_matches[1:]):  # skip 1st because its always invalid (match with itself)
            # omatch: cv2.DMatch
            query_id = omatch.queryIdx
            train_id = omatch.trainIdx
            img_id1 = get_img_id_from_kp_id(query_id, kp_ranges)
            img_id2 = get_img_id_from_kp_id(train_id, kp_ranges)
            if img_id1 == img_id2:
                # skip invalid match
                continue
            l_matches.append(omatch)  
            match_count[img_id1][img_id2] += 1
            if len(l_matches) >= const.KNN_FOR_PANORAMA:  # we only want a max of KNN_FOR_PANORAMA matches per feature
                break
        processed_matches.append(l_matches)

    return processed_matches, match_count


def get_kps_ranges(orb_images: ORBImageList) -> np.ndarray:
    kp_ranges = np.array([0], dtype=np.uint32)  # indicated at which ID an image starts/end

    for oi in orb_images:
        n_kps = kp_ranges[-1] + oi.n_kps
        kp_ranges = np.append(kp_ranges, n_kps)
    
    return kp_ranges


def get_img_id_from_kp_id(kp_id1: int, kp_ranges: np.ndarray) -> int:
    return np.searchsorted(kp_ranges, kp_id1, side='right') - 1


def assign_matches_to_orb_images(orb_imags: list[ORBImage], matches: KnnMatchesList):
    lower = 0
    for oi in orb_imags:
        upper = lower + oi.n_kps
        oi.set_matches(matches[lower:upper])
        lower = upper


def get_best_image_matches(match_counts: np.ndarray, img_id):
    n_images = match_counts.shape[0]
    M = const.M_CANDIDATE_IMAGES
    M = min(M, n_images-1)
    counts = match_counts[img_id]
    ranked_img_ids = np.argpartition(counts, -M)[-M:]
    ranked_img_ids = ranked_img_ids[np.argsort(counts[ranked_img_ids])[::-1]]
    return ranked_img_ids


def count_features_in_overlap(orb_img1: ORBImage, orb_img2: ORBImage, H: np.ndarray) -> int:
    if orb_img1.kps is None:
        return 0

    kps1 = orb_img1.kps
    h2, w2 = orb_img2.shape

    src_pts = np.empty((len(kps1), 1, 2), dtype=np.float32)
    for i, kp in enumerate(kps1):
        src_pts[i, 0, 0] = kp.pt[0]
        src_pts[i, 0, 1] = kp.pt[1]

    dst_pts = cv2.perspectiveTransform(src_pts, H)

    count = 0
    for i in range(len(kps1)):
        px, py = dst_pts[i, 0]
        if 0 <= px <= w2 and 0 <= py <= h2:
            count += 1

    return count

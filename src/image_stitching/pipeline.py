import cv2
import numpy as np
from typing import TypeAlias

from core.types import KnnMatches, MatchGroup, KnnMatchesList, MatchList, ORBDescriptor
from classes.ORBImage import ORBImage 

# Types
ORBImageList: TypeAlias = list[ORBImage]

def match_orb_images(orb_images: ORBImageList, k: int):
    descriptors = combine_descriptors(orb_images)
    matches = get_matches(descriptors, k)
    return matches


def combine_descriptors(orb_images: ORBImageList):
    combined_list = []

    for orb_image in orb_images:
        descriptors = orb_image.descriptors

        if descriptors is not None and len(descriptors) > 0:
            combined_list.append(descriptors)

    if not combined_list:
        return np.empty((0, 32), dtype=np.uint8)

    return np.vstack(combined_list)


def get_matches(descriptors: np.ndarray, k: int):
    # Define FLANN parameters and match descriptors
    # from https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
    FLANN_INDEX_LSH = 6
    index_params= dict(algorithm = FLANN_INDEX_LSH,
                    table_number = 6, # 12
                    key_size = 12,     # 20
                    multi_probe_level = 1) #2
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors, descriptors, k)

    return matches


def process_matches(orb_images: ORBImageList, matches: KnnMatches):
    kps_ranges = get_kps_ranges(orb_images)


def get_kps_ranges(orb_images: ORBImageList):
    kp_ranges = np.array([0], dtype=np.uint32)  # indicated at which ID an image starts/end

    for oi in orb_images:
        n_kps = kp_ranges[-1] + len(oi.kps)
        kp_ranges = np.append(kp_ranges, n_kps)
    
    return kp_ranges


def remove_all_invalid_matches(all_matches: KnnMatches, kp_ranges: np.ndarray) -> KnnMatchesList:
    valid_matches = []

    def is_valid_match(kp_id1: int, kp_id2: int):
        img_id1 = np.searchsorted(kp_ranges, kp_id1, side='right') - 1
        img_id2 = np.searchsorted(kp_ranges, kp_id2, side='right') - 1
        return img_id1 != img_id2
        
    for t_matches in all_matches:
        # t_matches: tuple of cv2.DMatch
        l_matches = []
        for omatch in t_matches[1:]:  # skip 1st because its always invalid (match with itself)
            # omatch: cv2.DMatch
            query_id = omatch.queryIdx
            train_id = omatch.trainIdx
            if not is_valid_match(query_id, train_id):
                continue
            l_matches.append(omatch)
        valid_matches.append(l_matches)
    
    return valid_matches
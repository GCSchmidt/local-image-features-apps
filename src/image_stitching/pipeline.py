import cv2
import numpy as np
from typing import TypeAlias
from shapely.geometry import Polygon, Point
from shapely import contains_xy

import core.constants as const
from core.types import KnnMatches, KnnMatchesList, ORBDescriptors, MatchList
from core.enums import MatchCountStrategy as MCS
from classes.ORBImage import ORBImage
from classes.FeatureImage import FeatureImage


########################
# Types
########################
ORBImageList: TypeAlias = list[ORBImage]
FeatureImageList: TypeAlias = list[FeatureImage]


########################
# Matching Images
########################
def match_orb_images(orb_images: ORBImageList, k: int) -> KnnMatches:
    descriptors = combine_descriptors(orb_images)
    matches = match_features(descriptors, descriptors, k)
    return matches


def match_images(images: FeatureImageList, k: int) -> KnnMatches:
    descriptors = combine_descriptors(images)
    return match_features(descriptors, descriptors, k)


def combine_descriptors(images: FeatureImageList) -> np.ndarray:
    combined_list = []

    for img in images:
        descriptors = img.descriptors

        if descriptors is not None and len(descriptors) > 0:
            combined_list.append(descriptors)

    if not combined_list:
        first_dtype = np.uint8 if isinstance(images[0], ORBImage) else np.float32
        first_dim = 32 if isinstance(images[0], ORBImage) else 128
        return np.empty((0, first_dim), dtype=first_dtype)

    return np.vstack(combined_list)


def match_features(descriptors1: np.ndarray, descriptors2: np.ndarray, k: int) -> KnnMatches:
    if descriptors1.dtype == np.uint8:
        # ORB features
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
    else:
        # SIFT features
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    return matcher.knnMatch(descriptors1, descriptors2, k)


def process_matches(images: FeatureImageList, all_matches: KnnMatches):
    processed_matches = []
    
    n_images = len(images)
    
    match_count = np.zeros((n_images, n_images))
    
    kp_ranges = get_kps_ranges(images)

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


def keep_relevant_matches(matchlist: MatchList, img_id: int, kp_ranges: np.ndarray) -> MatchList:
    reduced_matchlist = []
    for dmatch in matchlist:
        train_kp_id = dmatch.trainIdx
        img_id_of_match = get_img_id_from_kp_id(train_kp_id, kp_ranges)
        if img_id_of_match == img_id:
            reduced_matchlist.append(dmatch)
    return reduced_matchlist 


def is_good_match(matchlist: MatchList) -> bool:
    if len(matchlist) == 1:
        return True

    distance_best = matchlist[0].distance
    distance_second = matchlist[1].distance    
    
    return (distance_best < const.GOOD_MATCH_RATIO * distance_second)
        

def get_kps_ranges(images: FeatureImageList) -> np.ndarray:
    kp_ranges = np.array([0], dtype=np.uint32)

    for img in images:
        n_kps = kp_ranges[-1] + img.n_kps
        kp_ranges = np.append(kp_ranges, n_kps)
    
    return kp_ranges


def get_img_id_from_kp_id(kp_id1: int, kp_ranges: np.ndarray) -> int:
    return np.searchsorted(kp_ranges, kp_id1, side='right') - 1


def assign_matches_to_images(images: FeatureImageList, matches: KnnMatchesList):
    lower = 0
    for img in images:
        upper = lower + img.n_kps
        img.set_matches(matches[lower:upper])
        lower = upper


def get_best_image_matches(match_counts: np.ndarray, img_id):
    n_images = match_counts.shape[0]
    M = const.M_CANDIDATE_IMAGES
    M = min(M, n_images-1)
    counts = match_counts[img_id]
    ranked_img_ids = np.argpartition(counts, -M)[-M:]
    ranked_img_ids = ranked_img_ids[np.argsort(counts[ranked_img_ids])[::-1]]
    return ranked_img_ids


def count_features_in_overlap(img1: FeatureImage, img2: FeatureImage, H: np.ndarray) -> int:
    if img1.kps is None:
        return 0
    
    if img2.kps is None:
        return 0

    kps1 = img1.kps
    kps2 = img2.kps

    pts_1_1 = np.empty((len(kps1), 1, 2), dtype=np.float32)
    for i, kp in enumerate(kps1):
        pts_1_1[i, 0, 0] = kp.pt[0]
        pts_1_1[i, 0, 1] = kp.pt[1]
    pts_1_2 = cv2.perspectiveTransform(pts_1_1, H)

    pts_2_2 = np.empty((len(kps2), 1, 2), dtype=np.float32)
    for i, kp in enumerate(kps2):
        pts_2_2[i, 0, 0] = kp.pt[0]
        pts_2_2[i, 0, 1] = kp.pt[1]

    pts_1_2 = pts_1_2.reshape(-1, 2)
    pts_2_2 = pts_2_2.reshape(-1, 2)
    points = pts_1_2

    overlap_poly = compute_overlap_region(img1.shape, img2.shape, H)
    
    count = count_points_in_region(points, overlap_poly)

    return count


def compute_overlap_region(img1_shape, img2_shape, H):
    h1, w1 = img1_shape
    h2, w2 = img2_shape

    corners1 = np.array([[0, 0], [w1, 0], [w1, h1], [0, h1]], dtype=np.float32)
    corners1_projected = cv2.perspectiveTransform(corners1.reshape(-1, 1, 2), H).reshape(-1, 2)

    corners2 = np.array([[0, 0], [w2, 0], [w2, h2], [0, h2]], dtype=np.float32)
    
    poly1 = Polygon(corners1_projected)
    poly2 = Polygon(corners2)

    if not poly1.is_valid:
        poly1 = poly1.buffer(0)
    if not poly2.is_valid:
        poly2 = poly2.buffer(0)

    intersection = poly1.intersection(poly2)

    if not intersection.is_valid:
        intersection = intersection.buffer(0)
    if intersection.is_empty:
        return Polygon()

    return intersection


def count_points_in_region(points, overlap_polygon):
    if overlap_polygon.is_empty or not overlap_polygon.is_valid:
        return 0
    return int(contains_xy(overlap_polygon, points[:, 0], points[:, 1]).sum())
import cv2
import numpy as np
from typing import TypeAlias
from shapely.geometry import Polygon, Point
from shapely import contains_xy
import logging
import core.constants as const
from core.types import KnnMatches, KnnMatchesList, ORBDescriptors, MatchList
from core.enums import MatchCountStrategy as MCS
from image_stitching.panograph import ImageConnection, PanoGraph
from classes.orb_image import ORBImage
from classes.feature_image import FeatureImage


logger = logging.getLogger("image_stitching")


########################
# Types
########################
ORBImageList: TypeAlias = list[ORBImage]
FeatureImageList: TypeAlias = list[FeatureImage]


########################
# Matching Images
########################
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


def get_good_matches(img1: FeatureImage, img2: FeatureImage,):
    matched_pairs: list[tuple[int, int]] = []
    img2_kp_range = img2.get_kp_range_from_matches()
    for kp_id1 in range(img1.n_kps):
        matchlist = keep_relevant_matches(img1.matches[kp_id1], img2_kp_range)
        if (matchlist) and is_good_match(matchlist):
            train_kp_id = matchlist[0].trainIdx
            matched_kp_id2 = train_kp_id - img2_kp_range[0]
            matched_pairs.append((kp_id1, matched_kp_id2))
    return matched_pairs


def keep_relevant_matches(matchlist: MatchList, kp_range: tuple[int, int]) -> MatchList:
    reduced_matchlist = []
    for dmatch in matchlist:
        train_kp_id = dmatch.trainIdx
        if train_kp_id >= kp_range[0] and train_kp_id <= kp_range[1]:
            reduced_matchlist.append(dmatch)
    return reduced_matchlist 


def is_good_match(matchlist: MatchList) -> bool:
    if len(matchlist) == 1:
        return True

    distance_best = matchlist[0].distance
    distance_second = matchlist[1].distance    
    
    return (distance_best < const.GOOD_MATCH_RATIO * distance_second)
        

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


########################
# Feature Utils
########################
def get_kps_ranges(images: FeatureImageList) -> np.ndarray:
    kp_ranges = np.array([0], dtype=np.uint32)

    for img in images:
        n_kps = kp_ranges[-1] + img.n_kps
        kp_ranges = np.append(kp_ranges, n_kps)
    
    return kp_ranges


def get_img_kp_range(img_id: int, kp_ranges: np.ndarray) -> tuple[int, int]:
    return (kp_ranges[img_id], kp_ranges[img_id+1]) 


def get_img_id_from_kp_id(kp_id1: int, kp_ranges: np.ndarray) -> int:
    return np.searchsorted(kp_ranges, kp_id1, side='right') - 1


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

########################
# Image Connections
########################
def verify_image_connection(ic: ImageConnection) -> bool:
    if ic.n_inliers < const.MIN_INLIERS:
        return False
    if ic.n_overlap < const.MIN_INLIERS:
        return False
    
    return (1 - ic.weight) > const.INLIER_THRESHOLD


def _resize_image(img: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return img
    scale = max_dim / longest
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


########################
# Panorama Rendering
########################
def render_panorama(
    image_paths: list[str],
    homographies: dict[int, np.ndarray],
    output_path: str = const.OUTPUT_DIR + "panorama.jpg",
) -> str:
    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")

    for img_id, H in homographies.items():
        img = cv2.imread(image_paths[img_id], cv2.IMREAD_COLOR)
        img = _resize_image(img, const.MAX_IMAGE_DIM)
        h, w = img.shape[:2]

        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
        projected = np.empty((4, 2), dtype=np.float64)
        for i, (x, y) in enumerate(corners):
            pt = H @ np.array([x, y, 1.0])
            projected[i, 0] = pt[0] / pt[2]
            projected[i, 1] = pt[1] / pt[2]

        min_x = min(min_x, projected[:, 0].min())
        max_x = max(max_x, projected[:, 0].max())
        min_y = min(min_y, projected[:, 1].min())
        max_y = max(max_y, projected[:, 1].max())

    min_x = int(np.floor(min_x))
    min_y = int(np.floor(min_y))
    max_x = int(np.ceil(max_x))
    max_y = int(np.ceil(max_y))

    T = np.array([
        [1, 0, -min_x],
        [0, 1, -min_y],
        [0, 0, 1],
    ], dtype=np.float64)

    canvas_h = max_y - min_y
    canvas_w = max_x - min_x

    MAX_CANVAS_DIM = 8000
    if canvas_h > MAX_CANVAS_DIM or canvas_w > MAX_CANVAS_DIM:
        logger.warning(
            f"Projected canvas size ({canvas_w}x{canvas_h}) exceeds max dimension ({MAX_CANVAS_DIM}). "
            "This likely means the homographies describe a 360° loop that cannot be flattened."
        )
        raise ValueError(
            f"Canvas too large ({canvas_w}x{canvas_h}). "
            "Bundle adjustment may need stronger regularization or the scene is a full 360° loop."
        )

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float64)
    weight_map = np.zeros((canvas_h, canvas_w), dtype=np.float64)

    for img_id, H in homographies.items():
        H_final = (T @ H).astype(np.float32)
        img = cv2.imread(image_paths[img_id], cv2.IMREAD_COLOR)
        img = _resize_image(img, const.MAX_IMAGE_DIM).astype(np.float64)

        warped = cv2.warpPerspective(img, H_final, (canvas_w, canvas_h))
        mask = np.any(warped > 0, axis=2)

        canvas[mask] += warped[mask]
        weight_map[mask] += 1.0

    weight_map[weight_map == 0] = 1.0
    canvas /= weight_map[:, :, np.newaxis]

    final_img = np.clip(canvas, 0, 255).astype(np.uint8)

    gray = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)
    coords = cv2.findNonZero(gray)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        final_img = final_img[y:y + h, x:x + w]

    success = cv2.imwrite(output_path, final_img)
    if not success:
        raise RuntimeError(f"Failed to save panorama to {output_path}")

    return output_path
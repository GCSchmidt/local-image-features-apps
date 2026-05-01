import logging
import os
import pandas as pd
import cv2
import numpy as np
import networkx as nx
from enum import Enum
from itertools import compress
from dataclasses import dataclass
from utils import file_utils


MIN_MATCH_COUNT = 10


# Create a logger for this module
logger = logging.getLogger("image_stitching")


class ValidConnection(Enum):
    """
    Indicates whether a connection between a pair of images has been verified with inliers of the homography.
    """
    UNKNOWN = -1
    NOTCONNECTED = 0
    CONNECTED = 1


class HomographyReference(Enum):
    """
    Wether homography should map from/to the base image 
    """
    TOBASE = 0
    FROMBASE = 1


@dataclass
class ImageConnection:
    weight: float  # 1 - percentage of matches 
    inliers: int  # number of inliers
    valid_connection: ValidConnection
    homography: np.ndarray  # homography describing transfrom from reference to target
    reference: int  # id of base image
    target: int  # id of target image


class ImageProcessor():
    
    @staticmethod
    def reduce_res(image: np.ndarray, max_dimension: int = 500) -> np.ndarray:
        """
        Scales the image down such that the its longest dimension is reduced to the desired maximum length. 

        Args:
            image (np.ndarray): image to rescale
            max_dimension (int, optional): max length of width or height of the image. Defaults to 1_000.
        """
        h, w = image.shape[:2]

        if h > w and h > max_dimension:
            scale = max_dimension / h
        elif w > h and w > max_dimension:
            scale = max_dimension / w
        else:
            return image
        
        resized_image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)     
        resized_image = cv2.GaussianBlur(resized_image, (3, 3), 0)  # denoise

        return image


class StitchedImage():
    def __init__(self, img_paths: list[str], homographies: dict[int, np.ndarray],
                 base_id: int, max_output_size: int = 4096) -> None:
        self.img_paths = img_paths
        self.homographies = homographies
        self.base_id = base_id
        self.max_output_size = max_output_size
        self.scale_factor = self._compute_scale_factor()
        self.final_img: np.ndarray
        self.canvas: np.ndarray
        self.T: np.ndarray
        self.x_min: int
        self.y_min: int

    def _compute_scale_factor(self) -> float:
        n = len(self.img_paths)
        scale = max(0.3, 1.0 - (n - 1) * 0.1)
        return scale

    def stitch_images(self):
        self._create_canvas()
        self._place_base_image()
        for img_id in self.homographies:
            if img_id != self.base_id:
                self._warp_image(img_id)
        self._crop_to_content()
        self._downsample_if_needed()

    def _create_canvas(self):
        all_corners = []
        for img_id, H in self.homographies.items():
            img = cv2.imread(self.img_paths[img_id], cv2.IMREAD_COLOR_BGR)
            h, w = img.shape[:2]
            h_scaled, w_scaled = int(h * self.scale_factor), int(w * self.scale_factor)
            H_scaled = H.copy().astype(np.float64)
            H_scaled[0, 2] *= self.scale_factor
            H_scaled[1, 2] *= self.scale_factor
            corners = np.array([[0, 0], [w_scaled, 0], [w_scaled, h_scaled], [0, h_scaled]],
                               dtype=np.float32).reshape(-1, 1, 2)
            warped = cv2.perspectiveTransform(corners, H_scaled.astype(np.float32)).reshape(-1, 2)
            all_corners.append(warped)
        all_corners = np.vstack(all_corners)
        x_min, y_min = np.floor(all_corners.min(axis=0)).astype(int)
        x_max, y_max = np.ceil(all_corners.max(axis=0)).astype(int)
        self.x_min, self.y_min = x_min, y_min
        self.T = np.array([[1, 0, -x_min],
                           [0, 1, -y_min],
                           [0, 0, 1]], dtype=np.float64)
        canvas_h = y_max - y_min
        canvas_w = x_max - x_min
        self.canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    def _place_base_image(self):
        base_img = cv2.imread(self.img_paths[self.base_id], cv2.IMREAD_COLOR_BGR)
        base_img = cv2.resize(base_img,
                              None,
                              fx=self.scale_factor,
                              fy=self.scale_factor,
                              interpolation=cv2.INTER_AREA
                              )
        base_H = self.T @ self.homographies[self.base_id]
        base_H = base_H.copy().astype(np.float64)
        base_H[0, 2] *= self.scale_factor
        base_H[1, 2] *= self.scale_factor
        warped = cv2.warpPerspective(base_img, base_H.astype(np.float32),
                                     (self.canvas.shape[1], self.canvas.shape[0]))
        self.canvas = warped

    def _warp_image(self, img_id: int):
        img = cv2.imread(self.img_paths[img_id], cv2.IMREAD_COLOR_BGR)
        img = cv2.resize(img, None, fx=self.scale_factor, fy=self.scale_factor,
                        interpolation=cv2.INTER_AREA)
        H_total = self.T @ self.homographies[img_id]
        H_total = H_total.copy().astype(np.float64)
        H_total[0, 2] *= self.scale_factor
        H_total[1, 2] *= self.scale_factor
        warped = cv2.warpPerspective(img, H_total.astype(np.float32),
                                     (self.canvas.shape[1], self.canvas.shape[0]))
        mask = (warped > 0)
        self.canvas[mask] = warped[mask]

    def _crop_to_content(self):
        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        coords = cv2.findNonZero(gray)
        x, y, w, h = cv2.boundingRect(coords)
        self.canvas = self.canvas[y:y+h, x:x+w]
        self.final_img = self.canvas

    def _downsample_if_needed(self):
        h, w = self.final_img.shape[:2]
        max_dim = max(h, w)
        if max_dim <= self.max_output_size:
            return
        scale = self.max_output_size / max_dim
        new_w = int(w * scale)
        new_h = int(h * scale)
        self.final_img = cv2.resize(self.final_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def display(self):
        cv2.imshow('Stitched Image', self.final_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save(self):
        output_path = os.path.join(file_utils.OUTPUT_PATH, r"panorama.jpg")
        success = cv2.imwrite(output_path, self.final_img)
        return success


class PanoramaGraph():

    def __init__(self, match_threshold) -> None:
        self.graph = nx.Graph()
        self.match_threshold = match_threshold  # ratio of matches needed to be possible connection       
        self.fully_connected = False
        self.base_node = 0

    def initialize_graph(self, match_ratio: np.ndarray):
        """_summary_

        Args:
            match_ratio (np.ndarray): matrix specifying ratio of matches between images pairs
            threshold (float): ratio needed for a possible connection / edge 
        """
        n_images = match_ratio.shape[0]
        for i in range(n_images):
            for j in range(i+1, n_images):
                best_ratio = max(match_ratio[i, j], match_ratio[j, i]) 
                if best_ratio > self.match_threshold:
                    inverse_ratio = 1 - best_ratio
                    ic = ImageConnection(
                        weight=inverse_ratio,
                        inliers=0,
                        valid_connection=ValidConnection.UNKNOWN,
                        homography=np.identity(3),
                        reference=i,
                        target=j
                        )
                    self.graph.add_edge(i, j, weight=inverse_ratio, data=ic)

        self.set_fully_connected()
        self.get_base_node()

    def set_fully_connected(self):
        self.fully_connected = nx.is_connected(self.graph)

    def get_base_node(self):
        closeness = nx.closeness_centrality(self.graph)
        self.base_node = max(closeness, key=closeness.get)

    def get_k_closest_neighbors(self, node, k=3):
        neighbors = self.graph[node]  # adjacency dict
        sorted_neighbors = sorted(
            neighbors.items(),
            key=lambda x: x[1]['weight']  # sort by weight
        )
        return [n for n, _ in sorted_neighbors[:k]]
    
    def remove_invalid_edges(self) -> None:
        """
        Remove all edges that are not ValidConnection.CONNECTED
        """
        edges_to_remove = []

        for u, v, attr in self.graph.edges(data=True):
            ic = attr.get("data", None)

            if ic is None or ic.valid_connection != ValidConnection.CONNECTED:
                edges_to_remove.append((u, v))

        self.graph.remove_edges_from(edges_to_remove)
        self.get_base_node()

    def get_minimum_spanning_tree(self):
        self.graph = nx.minimum_spanning_tree(self.graph)
        self.get_base_node()

    def keep_largest_connected_group(self):
        largest_nodes = max(nx.connected_components(self.graph), key=len)
        nodes_to_remove = set(self.graph.nodes) - largest_nodes
        self.graph.remove_nodes_from(nodes_to_remove)
        self.get_base_node()

    def get_homographies(self, reference: HomographyReference = HomographyReference.TOBASE) -> dict[int, np.ndarray]:
        """
        Calculate all homographies to/from base image.
        Homographies are composed using BFS (shortest path).
        """
        from collections import deque

        homographies: dict[int, np.ndarray] = {self.base_node: np.eye(3)}
        queue = deque([self.base_node])

        while queue:
            current = queue.popleft()
            H_1_0 = homographies[current]

            for neighbor in self.graph[current]:
                if neighbor not in homographies:
                    ic = self.graph[current][neighbor]["data"]
                    # Get H from neighbor to current
                    H_2_1 = ic.homography
                    if ic.reference == current:
                        H_2_1 = np.linalg.inv(H_2_1)

                    H_2_0 = H_1_0 @ H_2_1
                    H_2_0 /= H_2_0[2, 2]
                    homographies[neighbor] = H_2_0
                    queue.append(neighbor)

        if reference == HomographyReference.FROMBASE:
            for img_id in homographies:
                homographies[img_id] = np.linalg.inv(homographies[img_id])

        return homographies

    def get_homography_from_to(self, target_id: int, reference: HomographyReference = HomographyReference.TOBASE) -> np.ndarray:
        """
        Calculates the homogrpahy to/from base image from/to target image

        Args:
            target_id (int): id of target image
            reference (HomographyReference): TOBASE or FROMBASE

        Returns:
            np.ndarray: homography
        """
        
        path = nx.shortest_path(self.graph, source=self.base_node, target=target_id)

        H_total = np.eye(3)

        for ref, target in zip(path[:-1], path[1:]):
            ic = self.graph[ref][target]["data"]
            H = ic.homography
            H_ref = ic.reference
            # Check direction
            if ref == H_ref:
                # default want target to base
                # reverse direction → invert homography
                H = np.linalg.inv(H)

            # Compose transformations
            H_total = H @ H_total

        if reference == HomographyReference.FROMBASE:
            H_total = np.linalg.inv(H_total)

        return H_total

    def get_image_connection(self, node1: int, node2: int) -> ImageConnection:
        ic = self.graph[node1][node2]["data"]
        return ic
    
    def set_image_connection(self, node1: int, node2: int, ic: ImageConnection) -> None:
        self.graph[node1][node2]["weight"] = ic.weight
        self.graph[node1][node2]["data"] = ic
        return ic
    

class Panorama():

    def __init__(self, path) -> None:
        self.folder_path = path
        self.image_files = file_utils.get_image_files(path)
        self.matching_df: pd.DataFrame
        self.img_id_bounds: np.ndarray
        self.K = min(len(self.image_files), 8)  # find K-nearest neighbours for each feature
        M_min = max(1, len(self.image_files)-1)
        self.M = min(M_min, 3)  # max number of images to verify connection
        self.connection_threshold = 1 / (len(self.image_files) + 1)  # ration of matches between 2 images, needed to check for possible connection
        self.inlier_ratio_threshold = 0.20 # inlier ratio needed to confirm a connection between images (inliers / number of good matches)
        self.inlier_count_threshold = 50  # inlier ratio needed to confirm a connection between images (inliers / number of good matches)
        self.SI: StitchedImage
        self.PG = PanoramaGraph(self.connection_threshold)

    def select_images(self, indices: list[int]):
        """
        Reduce self.image_files to a selected few of image determined by indices

        Args:
            indices (set[int]): ids of images to keep (range from 0 to len(self.image_files))
        """
        self.image_files = [self.image_files[i] for i in indices]

    def get_image_names(self) -> list[str]:
        out = [os.path.basename(path) for path in self.image_files] 
        return out

    def get_file_name(self, id) -> str:
        out = os.path.basename(self.image_files[id])
        return out

    def generate_panorama(self):
        self.generate_graph()
        self.stitch()

    def generate_graph(self):
        kps = self.detect()
        _ = self.match_all_images()
        log_msg = self.get_matching_stats_string()
        logger.debug(log_msg)
        match_ratios = self.match_ratio_matrix()
        self.PG.initialize_graph(match_ratios)
        self.establish_connections(kps)
        self.PG.get_minimum_spanning_tree()

    def detect(self) -> tuple[cv2.KeyPoint]:
        """
        Detects all keypoints in all the images and creates self.matching_df.

        Returns:
            tuple: tuple of OpenCV KeyPoint Objs
        """
        orb = cv2.ORB_create()
        self.img_id_bounds = np.array([0], dtype=np.uint32)  # indicates at which ID an image starts/end
        N = 0
        des = np.empty((0, 32), dtype=np.uint8)  # data type is crucial!
        img_ids = np.empty((0))
        kps = ()

        for img_id, img_path in enumerate(self.image_files): 
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            # img = ImageProcessor.reduce_res(img)
            kp_new, des_new = orb.detectAndCompute(img, None)  # type: ignore
            des = np.vstack((des, des_new))
            kps += kp_new
            N_new = len(des_new)
            N += N_new
            self.img_id_bounds = np.append(self.img_id_bounds, N)
            img_ids_new = np.full((N_new), img_id)
            img_ids = np.concatenate((img_ids, img_ids_new), axis=0)

        self.generate_matching_df(des, img_ids)

        return kps

    def generate_matching_df(self, des, img_ids) -> None:
        self.matching_df = pd.DataFrame(
            {"Descriptor": list(des),
             "ImgId": img_ids})
        # dont expect more than 2^8 / 255 images for the panorama
        convert_dict = {'ImgId': np.uint8}
        self.matching_df = self.matching_df.astype(convert_dict)

    def match_all_images(self) -> list[list[cv2.DMatch]]:
        """
        Adds 'Matched_Ids' collumns to self.matching_df, which contains ids 
        of matched descriptors.
        """
        # Define FLANN parameters and match descriptors
        # from https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=6,  # 12
                            key_size=12,     # 20
                            multi_probe_level=1)  # 2
        search_params = dict(checks=50)   # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        des = np.stack(list(self.matching_df["Descriptor"]))
        K = self.K + 1  # first match is always invalid
        matches = flann.knnMatch(des, des, k=K)
        distance_col = list()  # distances of match column 
        # create a list version of matches
        # and create Matched_Ids col.
        matches_list, matched_col = list(), list()

        for i, m_tuple in enumerate(matches):
            m_list = list(m_tuple[1:])  # 1st match is always the feature matches with itself
            matches_list.append(m_list)
            train_idx = []
            distances = []
            for m in m_list:
                train_idx.append(m.trainIdx)
                distances.append(m.distance)
            matched_col.append(np.array(train_idx))
            distance_col.append(np.array(distances))

        self.matching_df['Matched_Ids'] = matched_col
        self.matching_df['Distances'] = distance_col
        self.add_matched_imgids_col()
        # remove (invalid) matches from the same image, also modifies matches_list
        self.matching_df[["Matched_Ids", "Matched_ImgIds", "Distances"]] = self.matching_df.apply(self.remove_invalid_matches, axis=1, args=(matches_list,))
        return matches_list

    def keep_top_n_matches(self, n):
        """
        Reduce entries in "Matched_Ids", "Matched_ImgIds" and "Distances" columns of self.matching_df.
        Reduces to length of n.
        """
        for col in ["Matched_Ids", "Matched_ImgIds", "Distances"]:
            self.matching_df[col] = self.matching_df[col].map(lambda x: x[:n])

    def match_two_images(self, img_id1: int, img_id2: int, K):
        """
        Generate a list keypoint id pairs identifying which keypoints from the 2 images are matched.

        Args:
            img_id1 (int): id of reference image
            img_id2 (int): id of traget image

        Returns:
            list[tuple[int, int]]: pairs of matched keypoints, represented by their ids/indices.
        """

        # Define FLANN parameters and match descriptors
        # from https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=6,  # 12
                            key_size=12,     # 20
                            multi_probe_level=1)  # 2
        search_params = dict(checks=50)   # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # get descriptors of kps 
        des_list = self.matching_df[(self.matching_df["ImgId"] == img_id1)]["Descriptor"]
        des1 = np.stack(des_list)
        des_list = self.matching_df[(self.matching_df["ImgId"] == img_id2)]["Descriptor"]
        des2 = np.stack(des_list)
        # match
        matches = flann.knnMatch(des1, des2, k=K)
        # create kp id pairs
        matched_id_pairs = []
        for m_tuple in matches:
            # N matches / tuples
            kp_id1 = self.img_id_bounds[img_id1] + m_tuple[0].queryIdx # kp id in img1
            for match in m_tuple:
                # match is cv2.DMatch obj
                kp_id2 = self.img_id_bounds[img_id2] + match.trainIdx # kp id in img2
                kp_id_pair = (kp_id1, kp_id2)
                matched_id_pairs.append(kp_id_pair)

        return matched_id_pairs, matches

    def add_matched_imgids_col(self) -> None:
        """
        Adds Matched_ImgIds column to self.matching_df.
        """
        self.matching_df['Matched_ImgIds'] = self.matching_df.apply(
            self.matched_ids_to_img_ids,
            axis=1
            )
      
    def descriptor_id_to_img_id(self, id) -> int:
        """
        Get the associated Image ID for the given Descriptor ID, using img_id_bounds.
        Returns:
            int: in range from 0 to length of bounds - 1
        """
        return np.searchsorted(self.img_id_bounds, id, side='right') - 1

    def get_number_of_features_in_img(self, id) -> int:
        """
        Gets the total number of features in image with id.
        Returns:
            int: total features
        """
        total = self.img_id_bounds[id+1] - self.img_id_bounds[id] 
        return total

    def matched_ids_to_img_ids(self, row) -> np.ndarray:
        """Generate the img ids from the row's matched keypoint ids

        Args:
            row: a row from self.matching_df

        Returns:
            np.ndarray: ids of images belonging to keypoints in 'Matched_Ids'
        """
        matched_ids = row['Matched_Ids']
        img_ids = [self.descriptor_id_to_img_id(id) for id in matched_ids]
        return np.array(img_ids)

    def remove_invalid_matches(self, row, matches_list: list) -> pd.Series:
        """
        Removes Ids from Matched_Ids that are the are from the same Image (ImgId).
        Also modifies the list matches_list passed as an argument.
        Args:
            row: a row from a pandas dataframe

        Returns:
            series: a series for containing the updated Matched_Ids and Matched_ImgIds, with invalid values removed.
        """
        index, matched_ids, img_ids, distances, imgId = int(row.name), row['Matched_Ids'], row['Matched_ImgIds'], row['Distances'], row['ImgId'] 
        indeces_to_remove = np.where(img_ids == imgId)
        mask = np.ones(len(matched_ids), dtype=bool)
        mask[indeces_to_remove] = False
        matched_ids = matched_ids[mask, ...]
        img_ids = img_ids[mask, ...]
        distances = distances[mask]
        matches_list[index] = list(compress(matches_list[index], mask))
        return pd.Series([matched_ids, img_ids, distances])

    def get_number_of_matches_per_image_pair(self) -> pd.DataFrame:
        """
        Generate a dataframe specifying the amount of matches between all image pairs.
        Returns:
            pd.DataFrame: collumns: ImgId (reference image id), 
            MatchedWith (target image id), 
            Count (number of matches)
        """
        temp_df = self.matching_df.explode('Matched_ImgIds')
        result = (
            temp_df
            .groupby(['ImgId', 'Matched_ImgIds'])
            .size()
            .reset_index(name='Count')
            .rename(columns={'Matched_ImgIds': 'MatchedWith'})
        )
        return result
    
    def get_number_of_good_matches_per_image_pair(self) -> pd.DataFrame:
        """
        Generate a dataframe specifying the amount of good matches between all image pairs.
        A good macth is where the 1st macths has a distance 0.8 smaller than the 2nd best. 
        Returns:
            pd.DataFrame: collumns: ImgId (reference image id), 
            MatchedWith (target image id), 
            Count (number of matches)
        """
        temp_df = self.matching_df[
            self.matching_df['Distances'].apply(
                lambda d: len(d) >= 2 and d[0] < 0.8 * d[1]
            )
        ]
        temp_df = temp_df.explode('Matched_ImgIds')
        result = (
            temp_df
            .groupby(['ImgId', 'Matched_ImgIds'])
            .size()
            .reset_index(name='Count')
            .rename(columns={'Matched_ImgIds': 'MatchedWith'})
        )
        return result
    
    def match_count_matrix(self) -> np.ndarray:
        """
        Create a matrix indicating the total good matches between images.
        Reference image id is row id.
        Target image id is column id.
        Returns:
            np.ndarray:
        """
        n_images = len(self.image_files)
        matrix = np.zeros((n_images, n_images))

        df = self.get_number_of_good_matches_per_image_pair()

        matrix[
            df["ImgId"].to_numpy(),
            df["MatchedWith"].to_numpy()
        ] = df["Count"].to_numpy()

        return matrix
        
    def match_ratio_matrix(self) -> np.ndarray:
        """
        Create a matrix indicating the ratio between total good matches between images.
        Reference image id is row id.
        Target image id is column id.
        Returns:
            np.ndarray:
        """
        n_images = len(self.image_files)
        matrix = self.match_count_matrix()
        for id in range(n_images):
            n_ref_features = self.get_number_of_features_in_img(id)
            matrix[id] = matrix[id] / n_ref_features
        return matrix
    
    def rank_image_connections(self) -> dict[int, list[int]]:
        temp_df = self.get_number_of_good_matches_per_image_pair()
        img_connections_ranked = (
            temp_df.sort_values(['ImgId', 'Count'], 
                                ascending=[True, False]).groupby('ImgId')['MatchedWith'].apply(list).to_dict()
            )
        return img_connections_ranked
    
    def get_matching_stats_string(self) ->str:
        """
        Generates the the number of matches (counts and ratio) between images
        """
        rankings = self.rank_image_connections()
        M_count = self.match_count_matrix()
        M_ratio = self.match_ratio_matrix()
        lines = []

        for id, arr in rankings.items():
            name = self.get_file_name(id)
            lines.append(f"{id} | {name}")
            for match_id in arr:
                count = M_count[id][match_id]
                ratio = M_ratio[id][match_id]
                lines.append(
                    f"\t{match_id} | {self.get_file_name(match_id)} | {count} | {ratio:.2f}"
                )
            lines.append("")

        return "\n".join(lines)
    
    def get_matching_stats_string2(self) ->str:
        """
        Generates the the number of matches (counts and ratio) between images
        """
        rankings = self.rank_image_connections2()
        M_count = self.match_count_matrix()
        M_ratio = self.match_ratio_matrix()
        lines = []

        for id, arr in rankings.items():
            name = self.get_file_name(id)
            lines.append(f"{id} | {name}")
            for match_id in arr:
                count = M_count[id][match_id]
                ratio = M_ratio[id][match_id]
                lines.append(
                    f"\t{match_id} | {self.get_file_name(match_id)} | {count} | {ratio:.2f}"
                )
            lines.append("")

        return "\n".join(lines)
     
    def get_image_pair_matches(self, img_id1: int, img_id2: int) -> list[tuple[int, int]]:
        """
        Generates list of keypoint id pairs, from 2 image IDs. The result
        specifies which keypoints form reference are matched to the target image.    

        Args:
            img_id1 (int): id of reference image
            img_id2 (int): id of target image

        Returns:
            list[tuple[int, int]]: Pairs of matched keypoints, 
            represneted by their index in self.matching_df
        """
        sub_matching_df = self.get_sub_matching_df(img_id1, img_id2)
        matches = []
        for row in sub_matching_df.itertuples(index=True):
            match_pair = self.generate_match_pair_from_row(row, img_id2)
            if match_pair is not None:
                matches.append(match_pair)
        return matches

    def get_sub_matching_df(self, img_id1: int, img_id2: int) -> pd.DataFrame:
        """
        Returns subset of self.matching_df, containing features that are 
        matched between 2 specific images, reference (img_id1) and 
        target (img_id2) image.

        Args:
            img_id1 (int): what 'ImgId' column should be equal to.
            img_id2 (int): what should be in 'Matched_ImgIds' column.

        Returns:
            pandas dataframe: subset of matching_df
        """
        sub_matching_df = self.matching_df[
            (self.matching_df["ImgId"] == img_id1) &
            (self.matching_df['Matched_ImgIds'].apply(lambda x: img_id2 in x))
            ]
        return sub_matching_df

    def generate_match_pair_from_row(self, row, img_id2: int) -> tuple[int, int]:
        """
        Generate the best mathced keypoint pair in a row of self.matching_df, 
        that include specific target image.

        Args:
            row: row from self.matching_df, using (itertuples(index=True):
            img_id2 (int): img id of target image

        Returns:
            list[tuple[int, int]]: Pairs of matched keypoints
        """
        kp_id_img1 = int(row.Index)  # id of feature in reference image 
        matched_ids = np.array(row.Matched_Ids)  # ids of matched feature
        img_ids = np.array(row.Matched_ImgIds)  # ids of images of matched features
        distances = np.array(row.Distances)  # distances of matches
        mask = img_ids == img_id2
        valid_ids = matched_ids[mask]  # reduce array with mask
        valid_distances = distances[mask]   
        kp_id_img2 = valid_ids[0]
        if len(valid_ids) > 1:
            is_good_match = valid_distances[0] < 0.8 * valid_distances[1] 
            if not is_good_match:
                return None
            
        matched_pair = (kp_id_img1, int(kp_id_img2))
        return matched_pair

    def stitch_2(self, img_id1: int, img_id2: int) -> None:
        """
        Stitches the 2 selected images togther.

        Args:
            img_id1 (int): id of image
            img_id2 (int): id of image
        """

        ic = self.PG.get_image_connection(img_id1, img_id2)
        ref = ic.reference
        base = ic.target 
        Hs = {0: np.eye(3), 1: ic.homography}  # H from target to base
        img_paths = [self.image_files[base], self.image_files[ref]]
        IS = StitchedImage(img_paths, Hs, 0)
        IS.stitch_images()
        IS.save()

    def stitch(self) -> None:
        """
        """
        Hs = self.PG.get_homographies()
        base_id = self.PG.base_node
        IS = StitchedImage(self.image_files, Hs, base_id)
        IS.stitch_images()
        IS.save()

    def get_homogrphy(self, kps: tuple[cv2.KeyPoint], matched_kp_ids: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the homography describing the transformation of keypoints in
        one image to their matches counterparts in another image

        Args:
            kps (tuple[cv2.KeyPoint]): tuple of all detected KeyPoints across all the images 
            matched_kp_ids (list[tuple[int, int]]): list of matched keypoints 

        Returns:
            tuple: a tuple containing:
                - H (np.ndarray)
                - mask (np.ndarray)
        """
        src = np.empty((len(matched_kp_ids),2), dtype=np.float32)  # coords of features from img_id1 
        dst = np.empty((len(matched_kp_ids),2), dtype=np.float32)  # coords of features from img_id2

        for i, id_pair in enumerate(matched_kp_ids):
            kp_id1, kp_id2 = id_pair  # get ids of features 
            kp1, kp2 = kps[kp_id1], kps[kp_id2]  # use ids of features to get their objs
            src[i, 0] = kp1.pt[0]  # x coord
            src[i, 1] = kp1.pt[1]  # y coord
            dst[i, 0] = kp2.pt[0]  # x coord
            dst[i, 1] = kp2.pt[1]  # y coord

        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

        return H, mask   # transfromion to go from target to reference, and mask of whether matched_kp_ids agree to H 

    def establish_connections(self, kps):
        n_images = len(self.image_files)
        for ref_id in range(n_images):
            logger.debug(f"Establishing Connection of img: {ref_id} AKA: {self.get_file_name(ref_id)}")
            target_ids = self.PG.get_k_closest_neighbors(ref_id, self.M)
            for target_id in target_ids:
                logger.debug(f"\tChecking Connection with img: {target_id} AKA: {self.get_file_name(target_id)}",)
                current_ic = self.PG.get_image_connection(ref_id, target_id)
                if current_ic.valid_connection is not ValidConnection.CONNECTED:
                    new_ic = self.get_connection_between(ref_id, target_id, kps)
                    if new_ic.valid_connection == ValidConnection.CONNECTED:
                        self.PG.set_image_connection(ref_id, target_id, new_ic)
                        logger.debug(f"\tCONNECTED with {new_ic.inliers} inliers and {1-new_ic.weight} ratio")
                    elif new_ic.valid_connection == ValidConnection.NOTCONNECTED:
                        self.PG.graph.remove_edge(ref_id, target_id)
                        logger.debug(f"\tNOT CONNECTED with {new_ic.inliers} inliers and {1-new_ic.weight} ratio")
                else:
                    logger.debug(f"\tAlready CONNECTED with {current_ic.inliers} inliers and {1-current_ic.weight} ratio")
        self.PG.remove_invalid_edges()

    def get_inlier_ratio(self, img_id1, n_inliers: int) -> float:
        """
        Calculates the ratio of inlier matches from the estimated homography between 2 images.
        Args:
            img_id1 (int): img id
            n_inliers (int): number of inliers

        Returns:
            float: inlier ratio 
        """
        n_features = self.get_number_of_features_in_img(img_id1)
        inlier_percentage = n_inliers / n_features
        return inlier_percentage

    def get_connection_between(self, img_id1, img_id2, kps) -> ImageConnection:
        # case 1 img_id1 is reference
        IC = self.get_connection_from_and_to(img_id1, img_id2, kps)
        if IC.valid_connection == ValidConnection.CONNECTED:
            return IC
        
        # case 2 img_id2 is reference
        IC = self.get_connection_from_and_to(img_id2, img_id1, kps)
        if IC.valid_connection == ValidConnection.CONNECTED:
            return IC
        
        IC = ImageConnection(
            weight=1,
            inliers=0,
            valid_connection=ValidConnection.NOTCONNECTED,
            homography=np.identity(3),
            reference=img_id1,
            target=img_id2
        )
        return IC

    def get_connection_from_and_to(self, img_id1, img_id2, kps) -> ImageConnection:
        matched_ids = self.get_image_pair_matches(img_id1, img_id2)
        logger.debug(f"\t\tnumber of good matches used for RANSAC: {len(matched_ids)}")
        H, mask = self.get_homogrphy(kps, matched_ids)
        n_inliers = np.count_nonzero(mask)
        inlier_ratio = n_inliers / len(matched_ids)
        connection_check = inlier_ratio > self.inlier_ratio_threshold
        connection_check &= n_inliers > self.inlier_count_threshold
        if connection_check:
            valid_connection = ValidConnection.CONNECTED
        else:
            valid_connection = ValidConnection.NOTCONNECTED
        IC = ImageConnection(
            weight=1 - inlier_ratio,
            inliers=n_inliers,
            valid_connection=valid_connection,
            homography=H,
            reference=img_id1,
            target=img_id2
        )
        return IC

    def get_image_sequence_from_graph(self) -> list[str]:
        """_summary_

        Returns:
            list[str]: _description_
        """
        # check if self.PG has nodes/edges

        if not self.PG.graph.has_node(0):
            raise Exception("Need to generate the graph first")

        nodes = list(nx.dfs_preorder_nodes(self.PG.graph))
        images = [os.path.basename(self.image_files[node]) for node in nodes]

        return images


class CVPanorama():

    def __init__(self, path, mode=0):
        self.folder_path = path
        self.image_files = file_utils.get_image_files(path)
        self.images = [cv2.imread(path) for path in self.image_files]
        self.mode = 0  # 0 = cv2.Stitcher_PANORAMA
        if mode == 1:
            # only other valid mode
            self.mode = 1  # 1= cv2.Stitcher_SCANS
        self.output_image: np.ndarray

    def generate_panorama(self) -> None:
        self.pre_processing()
        self.output_image = self.stitch_images()
        self.post_processing()
        output_path = os.path.join(file_utils.OUTPUT_PATH, r"cv_panorama.jpg")

        if self.output_image is None:
            print("Failed to save panorama")
        else:
            print(f"Saving panorama to {output_path}")
            cv2.imwrite(output_path, self.output_image)

    def stitch_images(self):
        stitcher = cv2.Stitcher_create(self.mode)
        status, panorama = stitcher.stitch(self.images)

        if status == cv2.Stitcher_OK:
            return panorama
        else:
            print("Error during stitching: ", status)
            return None

    def pre_processing(self):
        self.reduce_res()

    def reduce_res(self):
        max_width = 1000

        for i, img in enumerate(self.images):
            h, w = img.shape[:2]
            if w > max_width:
                scale = max_width / w
                img = cv2.resize(img, None, fx=scale, fy=scale)     
            # Optional: denoise
            img = cv2.GaussianBlur(img, (3, 3), 0)     
            self.images[i] = img

    def post_processing(self):
        if self.output_image is None:
            return
        self.crop_convex_hull()

    def crop_convex_hull(self):
        gray = cv2.cvtColor(self.output_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        biggest = max(cnts, key=cv2.contourArea)
        hull = cv2.convexHull(biggest)

        mask = np.zeros_like(gray)
        cv2.fillConvexPoly(mask, hull, 255)

        x, y, w, h = cv2.boundingRect(hull)
        self.output_image = self.output_image[y:y+h, x:x+w]


if __name__ == "__main__":
    pass

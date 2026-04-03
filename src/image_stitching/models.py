import logging
import os
import pandas as pd
import cv2
import numpy as np
import networkx as nx
from enum import Enum
from itertools import compress
from pathlib import Path
from dataclasses import dataclass
from utils import file_utils



MIN_MATCH_COUNT = 10


# Create a logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False

# Create a file handler (unique to this module)
log_path = os.path.join(file_utils.LOG_PATH, 'image_stitching/models.log')
file_handler = logging.FileHandler(log_path, mode='w')  # overwrite/replace the file 
file_handler.setLevel(logging.DEBUG)

# Optional formatter
formatter = logging.Formatter(
    '%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s\n%(message)s'
)
file_handler.setFormatter(formatter)

# Add the handler if not already added (to avoid duplicates)
if not logger.handlers:
    logger.addHandler(file_handler)


class ValidConnection(Enum):
    """
    Indicates whether a connection between a pair of images has been verified with inliers of the homography.
    """
    UNKNOWN = -1
    NOTCONNECTED = 0
    CONNECTED = 1


@dataclass
class ImageConnection:
    weight: float  # 1 - percentage of matches 
    inliers: int  # number of inliers
    valid_connection: ValidConnection
    homography: np.ndarray  # homography describing transfrom from image with lower index to higher
    reference: int  # id of base image
    target: int  # id of target image


class StitchedImage():
    """
    A class that is used to construct the panorama result / final image.
    """
    def __init__(self, img_paths) -> None:
        self.img_paths = img_paths
        N = len(self.img_paths)
        self.homographies = np.tile(np.identity(3), (N, 1, 1))
        self.final_img = cv2.imread(img_paths[0], cv2.IMREAD_COLOR_BGR)
        self.connected = set([0])

    def is_connected(self, img_id:int) -> bool:
        return (img_id in self.connected)

    def stitch_image(self, img_id: int):
        H = self.homographies[img_id]
        
        h1, w1 = self.final_img.shape[:2]

        new_img_path = self.img_paths[img_id]
        new_img = cv2.imread(new_img_path, cv2.IMREAD_COLOR_BGR)
        h2, w2 = new_img.shape[:2]

        # Corners of new_img being stitched
        corners_img2 = np.array([[0,0],[0,h2],[w2,h2],[w2,0]], dtype=np.float32).reshape(-1,1,2)
        # Warp corners into img1's frame using H
        warped_corners = cv2.perspectiveTransform(corners_img2, H)

        # Corners of final_img
        corners_final_img = np.array([[0,0],[0,h1],[w1,h1],[w1,0]], dtype=np.float32).reshape(-1,1,2)

        # All corners together to find canvas size
        all_corners = np.concatenate((warped_corners, corners_final_img), axis=0)
        [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

        # Translation matrix to shift everything into positive coordinates
        translation = np.array([[1,0,-xmin],[0,1,-ymin],[0,0,1]])

        final_img_copy = self.final_img.copy()
        # Warp new_img into canvas
        self.final_img = cv2.warpPerspective(new_img, translation @ H, (xmax - xmin, ymax - ymin))
        # Place the original/base final_img into canvas
        self.final_img[-ymin:h1 - ymin, -xmin:w1 - xmin] = final_img_copy

    def update_homography(self, img_id1: int, img_id2: int, H1_2: np.ndarray):
        """
        Use the transformation from img_id1 to img_id2, 
        to update the transform from base to img_id2
        Args:
            img_id1 (int): reference (image to connect)
            img_id2 (int): target image (image connected already connected)
            H1_2 (np.ndarray): homography from reference to target
        """
        if not self.is_connected(img_id2):
            logger.debug(f"Failed to update homography of Img {img_id1} with the homography of Img {img_id2}." +
                         f"Img {img_id2} not yet added to connected: {self.connected}")
            return
        logger.debug(f"Homography from Img {img_id1} to Img {img_id2}:\n{H1_2}")

        H2_0 = self.homographies[img_id2]  # TF from img ID 2 to base  
        H1_0 = H2_0 @ H1_2 # TF from img ID 1 to base  
        self.homographies[img_id1] = H1_0
        h_list = [str(x) for x in self.homographies]
        logger.debug("\n".join(h_list))
        self.connected.add(img_id1)
        logger.debug(f"Img {img_id1} connected via {img_id2}\nconnected set: {self.connected}")

    def stitch_images(self):
        # Load all images
        imgs = [cv2.imread(p) for p in self.img_paths]
        base = imgs[0]

        # Homographies (all should be image -> base frame)
        Hs = [np.eye(3)] + list(self.homographies[1:])

        # Compute canvas size in original coordinates
        all_corners = []
        for img, H in zip(imgs, Hs):
            h, w = img.shape[:2]
            corners = np.array([[0,0], [w,0], [w,h], [0,h]], dtype=np.float32)
            warped = cv2.perspectiveTransform(corners.reshape(-1,1,2), H)
            all_corners.append(warped.reshape(-1,2))
        all_corners = np.vstack(all_corners)

        x_min, y_min = np.floor(all_corners.min(axis=0)).astype(int)
        x_max, y_max = np.ceil(all_corners.max(axis=0)).astype(int)

        canvas_w, canvas_h = x_max - x_min, y_max - y_min

        # Scale factor
        MAX_DIM = 3000
        scale = min(1.0, MAX_DIM / max(canvas_w, canvas_h))

        # Scale images
        scaled_imgs = [cv2.resize(img, None, fx=scale, fy=scale) for img in imgs]

        # Correct homography scaling:  H_scaled = S * H * S⁻¹
        S = np.array([[scale, 0, 0],
                    [0, scale, 0],
                    [0, 0, 1]], dtype=np.float32)
        S_inv = np.linalg.inv(S)

        scaled_Hs = [S @ H @ S_inv for H in Hs]

        # Compute new canvas translation
        all_corners_scaled = []
        for img, H in zip(scaled_imgs, scaled_Hs):
            h, w = img.shape[:2]
            corners = np.array([[0,0], [w,0], [w,h], [0,h]], dtype=np.float32)
            warped = cv2.perspectiveTransform(corners.reshape(-1,1,2), H)
            all_corners_scaled.append(warped.reshape(-1,2))
        all_corners_scaled = np.vstack(all_corners_scaled)

        x_min, y_min = np.floor(all_corners_scaled.min(axis=0)).astype(int)
        x_max, y_max = np.ceil(all_corners_scaled.max(axis=0)).astype(int)

        tx, ty = -x_min, -y_min
        T = np.array([[1, 0, tx],
                    [0, 1, ty],
                    [0, 0, 1]], dtype=np.float32)

        # Create final canvas
        self.final_img = np.zeros((y_max - y_min, x_max - x_min, 3), dtype=np.uint8)

        # Warp each image
        for img, H in zip(scaled_imgs, scaled_Hs):
            Ht = T @ H
            cv2.warpPerspective(
                img, Ht,
                (self.final_img.shape[1], self.final_img.shape[0]),
                self.final_img,
                borderMode=cv2.BORDER_TRANSPARENT
            )
            
    def stitch_images2(self):
        def corners(img, H):
            h, w = img.shape[:2]
            pts = np.array([[0,0], [w,0], [w,h], [0,h]], np.float32).reshape(-1,1,2)
            return cv2.perspectiveTransform(pts, H).reshape(-1,2)
        
        images = [cv2.imread(path, cv2.IMREAD_COLOR_BGR) for path in self.img_paths]
        homographies = self.homographies

        all_pts = np.vstack([corners(img, H) for img, H in zip(images, homographies)])
        x_min, y_min = np.floor(all_pts.min(axis=0)).astype(int)
        x_max, y_max = np.ceil(all_pts.max(axis=0)).astype(int)

        trans = np.array([[1,0,-x_min],[0,1,-y_min],[0,0,1]], np.float64)
        width, height = x_max - x_min, y_max - y_min

        acc = np.zeros((height, width, 3), np.float32)
        count = np.zeros((height, width, 1), np.float32)

        for img, H in zip(images, homographies):
            Ht = trans @ H
            warped = cv2.warpPerspective(img, Ht, (width, height))
            mask = (warped > 0).astype(np.float32)
            acc += warped.astype(np.float32)
            count += mask[..., :1]

        self.final_img = np.divide(acc, np.maximum(count, 1), where=count>0).astype(np.uint8)

    def display(self):
        cv2.imshow('Sticthed Image', self.final_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def save(self):
        output_path = os.path.join(file_utils.OUTPUT_PATH, r"panorama.jpg")
        success = cv2.imwrite(output_path, self.final_img)


class PanoramaGraph():

    def __init__(self, match_threshold) -> None:
        self.graph = nx.Graph()
        self.match_threshold = match_threshold  # ration of matches needed to be possible connection       
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

    def get_minimum_spanning_tree(self):
        self.graph = nx.minimum_spanning_tree(self.graph)

    def get_homography_from_to(self, target_id: int) -> np.ndarray:
        """
        Calculates the homogrpahy from base image to target image

        Args:
            target_id (int): id of target image

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
            if ref != H_ref:
                # reverse direction → invert homography
                H = np.linalg.inv(H)

            # Compose transformations
            H_total = H @ H_total

        return H_total


class Panorama():

    def __init__(self, path) -> None:
        self.folder_path = path
        self.image_files = file_utils.get_image_files(path)
        self.matching_df: pd.DataFrame
        self.img_id_bounds: np.ndarray
        self.K = min(len(self.image_files), 5)  # find K-nearest neighbours for each feature
        min_n_imgs = max(1, len(self.image_files)-1)
        self.M = min(min_n_imgs, 3)  # number of images to check for connection
        self.connection_threshold = 0.25  # ration of matches between 2 images, needed to check for possible connection
        self.inlier_threshold = 0.15  # inlier ratio needed to confirm a connection between images
        self.SI: StitchedImage
        self.PG = PanoramaGraph(self.connection_threshold)

    def select_images(self, indices: list[int]):
        """
        Reduce self.image_files to a selected few of image determined by indices

        Args:
            indices (set[int]): ids of images to keep (range from 0 to len(self.image_files))
        """
        self.image_files = [self.image_files[i] for i in indices]
   
    def generate_panorama(self):
        self.generate_graph()
        self.stitch()

    def generate_graph(self):
        kps = self.detect()
        _ = self.match_all_images()
        self.keep_top_n_matches(self.K)
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
        self.img_id_bounds = np.array([0], dtype=np.uint32)  # indicated at which ID an image starts/end
        N = 0
        des = np.empty((0, 32), dtype=np.uint8)  # data type is crucial!
        img_ids = np.empty((0))
        kps = ()

        for img_id, img_path in enumerate(self.image_files): 
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
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
        K = self.K * 2 # so that we are sure to have more matches after reduction 
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
        self.matching_df[["Matched_Ids", "Matched_ImgIds"]] = self.matching_df.apply(self.remove_invalid_matches, axis=1, args=(matches_list,))
        return matches_list

    def keep_top_n_matches(self, n):
        """
        Reduce entries in "Matched_Ids", "Matched_ImgIds" and "Distances" columns of self.matching_df.
        Reduces to length of n.
        """
        for col in ["Matched_Ids", "Matched_ImgIds", "Distances"]:
            self.matching_df[col] = self.matching_df[col].map(lambda x: x[:n])

    def match_two_images(self, img_id1: int, img_id2: int, K) -> list[tuple[int, int]]:
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

        return matched_id_pairs

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
        index, matched_ids, img_ids, imgId = int(row.name), row['Matched_Ids'], row['Matched_ImgIds'], row['ImgId'] 
        indeces_to_remove = np.where(img_ids == imgId)
        # create mask to remove values
        mask = np.ones(len(matched_ids), dtype=bool)
        mask[indeces_to_remove] = False
        matched_ids = matched_ids[mask, ...]
        img_ids = img_ids[mask, ...]
        matches_list[index] = list(compress(matches_list[index], mask))
        return pd.Series([matched_ids, img_ids])

    def match_stats(self):
        # Get the counts of imag matches
        print(self.get_number_of_matches_per_image_pair())

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
    
    def match_count_matrix(self) -> np.ndarray:
        """
        Create a matrix indicating the total matches between images.
        Reference image id is row id.
        Target image id is column id.
        Returns:
            np.ndarray:
        """
        n_images = len(self.image_files)
        matrix = np.zeros((n_images, n_images))

        df = self.get_number_of_matches_per_image_pair()

        matrix[
            df["ImgId"].to_numpy(),
            df["MatchedWith"].to_numpy()
        ] = df["Count"].to_numpy()

        return matrix
        
    def match_ratio_matrix(self) -> np.ndarray:
        """
        Create a matrix indicating the ratio between total matches between images.
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
        temp_df = self.get_number_of_matches_per_image_pair()
        logger.debug(f"Matches Per Image Pair:\n {temp_df.to_string()}")
        img_connections_ranked = (
            temp_df.sort_values(['ImgId', 'Count'], 
                                ascending=[True, False]).groupby('ImgId')['MatchedWith'].apply(list).to_dict()
            )
        return img_connections_ranked

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
            matches.extend(self.generate_match_pairs_from_row(row, img_id2))
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

    def generate_match_pairs_from_row(self, row, img_id2: int) -> list[tuple[int, int]]:
        """
        Generate all mathced keypoint pairs in a row of self.matching_df, 
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
        mask = img_ids == img_id2
        valid_id = matched_ids[mask]  # reduce array with mask
        matched_pairs = [(kp_id_img1, int(kp_id_img2)) for kp_id_img2 in valid_id]
        return matched_pairs

    def stitch(self) -> None:
        """
        """
        pass


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
            target_ids = self.PG.get_k_closest_neighbors(ref_id, self.M)
            for target_id in target_ids:
                edge = self.PG.graph[ref_id][target_id]
                current_ic = edge['data']
                if current_ic.valid_connection is not ValidConnection.CONNECTED:
                    new_ic = self.get_connection_between(ref_id, target_id, kps)
                    if new_ic.valid_connection == ValidConnection.CONNECTED:
                        self.PG.graph[ref_id][target_id]['data'] = new_ic
                    elif new_ic.valid_connection == ValidConnection.NOTCONNECTED:
                        self.PG.graph.remove_edge(ref_id, target_id)
        # self.PG.remove_invalid_edges()

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
        IC = self.get_connection_from_and_to(img_id1, img_id2, kps)
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
        H, mask = self.get_homogrphy(kps, matched_ids)
        n_inliers = np.count_nonzero(mask)
        inlier_ratio = self.get_inlier_ratio(img_id1, n_inliers)
        if inlier_ratio > self.inlier_threshold:
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

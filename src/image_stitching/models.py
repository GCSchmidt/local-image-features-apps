import logging
import os
import pandas as pd
import cv2
import numpy as np
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


class Panorama():

    def __init__(self, path) -> None:
        self.folder_path = path
        self.image_files = file_utils.get_image_files(path)
        self.matching_df: pd.DataFrame
        self.img_id_bounds: np.ndarray
        self.K = 8  # find K-nearest neighbours during matching
        min_n_imgs = max(1, len(self.image_files)-1)
        self.M = min(min_n_imgs, 2)  # number of best matched images to use per image
        self.valid_connection_threshold = 0.25  # percentage of kps matched between images, to accept them to be connected
        self.SI: StitchedImage

    def select_images(self, indices: set[int]):
        """
        Reduce self.image_files to a selected few of image determined by 

        Args:
            indices (set[int]): ids of images to keep (range from 0 to len(self.image_files))
        """
        self.image_files = [self.image_files[i] for i in indices]
   
    def generate_panorama(self):
        self.SI = StitchedImage(self.image_files)
        kps = self.detect()
        _ = self.match_all_images()
        img_connections_ranked = self.rank_image_connections()
        formatted_data = "\n".join([f"{k}:{v}" for k, v in img_connections_ranked.items()])
        logger.debug(f"img_connections_ranked:\n{formatted_data}")
        self.stitch(img_connections_ranked, kps)

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
        matches = flann.knnMatch(des, des, k=self.K)
        # create a list version of matches
        # and create Matched_Ids col.
        matches_list, matched_col = list(), list()
        for i, m_tuple in enumerate(matches):
            m_list = list(m_tuple[1:])  # 1st match is always the feature matches with itself
            matches_list.append(m_list)
            matched_col.append(np.array([m.trainIdx for m in m_list]))

        self.matching_df['Matched_Ids'] = matched_col
        self.add_matched_imgids_col()
        # remove (invalid) matches from the same image, also modifies matches_list
        self.matching_df[["Matched_Ids", "Matched_ImgIds"]] = self.matching_df.apply(self.remove_invalid_matches, axis=1, args=(matches_list,))
        return matches_list

    def match_two_images(self, img_id1: int, img_id2: int) -> list[tuple[int, int]]:
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
        matches = flann.knnMatch(des1, des2, k=3)
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
            Count (nnumber of matches)
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

    def stitch(self, img_connections_ranked: dict, kps: tuple[cv2.KeyPoint]) -> None:
        """_summary_

        Args:
            img_connections_ranked (dict): _description_
            kps (tuple[cv2.KeyPoint]): _description_
        """
        for img_id1, img_ids in img_connections_ranked.items():
            if self.SI.is_connected(img_id1):
                # if we have a valid homography for this image, we can use it
                # to find the homographies of the images connected to it. 
                for img_id2 in img_ids[0:self.M]:
                    
                    # go through top self.M best connected images
                    if not self.SI.is_connected(img_id2):

                        logger.debug(f"Trying to connect Img {img_id2} via Img {img_id1}")

                        # only need to determine homography if the image is 
                        # not yet connected/had its homography calculated previously
                        
                        # get matched pairs of keypoints with

                        # option 1: might miss some keypint matches because limited to prevoius matching method
                        # matched_kp_ids = self.get_image_pair_matches(img_id2, img_id1)
                        
                        # option 2: redos matching, to get more possible matches
                        matched_kp_ids = self.match_two_images(img_id2, img_id1)
                        
                        H, mask = self.get_homogrphy(kps, matched_kp_ids)

                        connected_flag = self.determine_connection(img_id2, img_id1, mask)
                        if connected_flag:
                            logger.debug(f"{img_id1} and Img {img_id2} are connected")
                            self.SI.update_homography(img_id2, img_id1, H)
                        else:
                            logger.debug(f"{img_id1} and Img {img_id2} are not connected")
            else:
                # if we dont have a valid homography for this image, we can use the
                # the homography of an image connected to it. 
                for img_id2 in img_ids[0:self.M]:
                    
                    # go through top self.M best connected images
                    if self.SI.is_connected(img_id2):

                        logger.debug(f"Trying to connect Img {img_id1} via Img {img_id2}")

                        # need an image for which we already have a homography

                        # option 1: might miss some keypint matches because limited to prevoius matching method
                        # matched_kp_ids = self.get_image_pair_matches(img_id1, img_id2)

                        # option 2: redos matching, to get more possible matches
                        matched_kp_ids = self.match_two_images(img_id1, img_id2)

                        H, mask = self.get_homogrphy(kps, matched_kp_ids)
                        connected_flag = self.determine_connection(img_id1, img_id2, mask)
                        if connected_flag:
                            logger.debug(f"Img {img_id1} and Img {img_id2} are connected")
                            self.SI.update_homography(img_id1, img_id2, H)
                        else:
                            logger.debug(f"Img {img_id1} and Img {img_id2} are not connected")

        self.SI.stitch_images()
        self.SI.display()
        self.SI.save()

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



if __name__ == "__main__":
    pass

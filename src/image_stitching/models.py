import logging
import os
import pandas as pd
import cv2
import numpy as np
from itertools import compress
from pathlib import Path
from dataclasses import dataclass
from utils import file_utils

logger = logging.getLogger(__name__)
log_path = os.path.join(file_utils.LOG_PATH, r'image_stitching/models.log')
logging.basicConfig(filename=log_path, level=logging.DEBUG)
logger.info('Started')

MIN_MATCH_COUNT = 10

@dataclass
class MatchResult:
    img1: str
    img2: str
    n_feats1: int
    n_feats2: int
    n_inliers: int
    percent_in: float


class StitchedImage():
    """
    A class that is used to construct the panorama result / final image.
    """
    def __init__(self, img_paths) -> None:
        self.img_paths = img_paths
        N = len(self.img_paths)
        self.homographies = np.tile(np.identity(3), (N, 1, 1))
        self.final_img = cv2.imread(img_paths[0], cv2.IMREAD_COLOR_BGR)
        
    def add_image(self, img_id: int):
        H = self.homographies[img_id]
        
        h1, w1 = self.final_img.shape[:2]

        new_img_path = self.img_paths[img_id-1]
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

        final_img_copy = 0
        # Warp new_img into canvas
        self.final_img = cv2.warpPerspective(new_img, translation @ H, (xmax - xmin, ymax - ymin))
        # Place the original/base final_img into canvas
        self.final_img[-ymin:h1 - ymin, -xmin:w1 - xmin] = final_img_copy

    def update_homography(self, img_id1: int, img_id2: int, H1_2: np.ndarray):
        H2 = self.homographies[img_id2] # TF from img ID 2 to base  
        new_H = H2 @ H1_2
        self.homographies[img_id1] = new_H

    def display(self):
        cv2.imshow('Sticthed Image', self.final_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def save(self):
        output_path = os.path.join(file_utils.OUTPUT_PATH, r"stitched.jpg")
        success = cv2.imwrite(output_path, self.final_img)

class Panorama():

    def __init__(self, path) -> None:
        self.folder_path = path
        self.image_files = file_utils.get_image_files(path)
        self.matching_df: pd.DataFrame
        self.img_id_bounds: np.ndarray
        self.K = 5  # find K-nearest neighbours during matching
        self.M = 2  # number of best matched images to use per image
        self.SI = StitchedImage(self.image_files)

    def generate_panorama(self):
        kps = self.detect()
        logger.debug(f"type of kps from detect(): {type(kps)}, {type(kps[0])}")
        matches = self.match()
        logger.debug(f"type of matches from detect(): {type(matches)}, {type(matches[0])}, {type(matches[0][0])}")
        self.SI

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
            img_ids_new = np.full((N_new), img_id+1)
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

    def match(self) -> list[list[cv2.DMatch]]:
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
            int: in range from 1 to length of bounds
        """
        return np.searchsorted(self.img_id_bounds, id, side='right')

    def matched_ids_to_img_ids(self, row):
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
    
    def rank_image_connections(self):
        temp_df = self.get_number_of_matches_per_image_pair()
        image_connections_ranked = (
            temp_df.sort_values(['ImgId', 'Count'], 
                                ascending=[True, False]).groupby('ImgId')['MatchedWith'].apply(list).to_dict()
            )


    def determine_connections(self, img_id1: int, img_id2: int, kps: list[cv2.KeyPoint],):
        
        return H 

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
            (self.matching_df["ImgId"] == img_id1) and
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


if __name__ == "__main__":
    pass

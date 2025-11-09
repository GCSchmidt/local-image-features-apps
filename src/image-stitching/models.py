import pandas as pd
import cv2
import numpy as np
from itertools import compress
from pathlib import Path
from dataclasses import dataclass
from utils import file_utils


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
    def __init__(self, ref_image) -> None:
        pass


class Panorama():

    def __init__(self, path) -> None:
        self.folder_path = path
        self.image_files = file_utils.get_image_files(path)
        self.matching_df: pd.DataFrame
        self.img_id_bounds: np.ndarray
        self.K = 5  # find K-nearest neighbours during matching
        self.M = 2  # number of best matched images to use per image

    def generate_panorama(self):
        kps = self.detect()
        matches = self.match()

    def detect(self):
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

    def match(self) -> list:
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
        img_id_stats_df = self.matching_df.explode('Matched_ImgIds')
        result = (
            img_id_stats_df
            .groupby(['ImgId', 'Matched_ImgIds'])
            .size()
            .reset_index(name='Count')
            .rename(columns={'Matched_ImgIds': 'MatchedWith'})
        )
        print(result)

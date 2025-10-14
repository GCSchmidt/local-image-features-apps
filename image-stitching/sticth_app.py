import os
import sys
import re
import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import MatchResult

MIN_MATCH_COUNT = 10

def get_image_files(path: str) -> list:
    """Returns a list of image files inside path."""

    image_types = [".jpg", ".png", ".jpeg"]
    image_files = [""]*(len(os.listdir(path))+1)

    for entry in os.scandir(path):

        if not entry.is_file():
            continue

        filename, extension = os.path.splitext(entry)
        if extension.lower() not in image_types:
            continue

        result = re.search(r'\d+$', filename)
        if not result:
            continue

        index = int(result.group())

        image_files[index] = filename + extension
    
    image_files = list(filter(lambda item: item != "", image_files))

    return image_files

def match_features(img1_path: str, img2_path: str) -> MatchResult:
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)  # ref
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)  # target

    if img1 is None:
        raise ValueError(f"Could not load image: {img1_path}")
    if img2 is None:
        raise ValueError(f"Could not load image: {img2_path}")

    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(img1, None)  # type: ignore
    kp2, des2 = orb.detectAndCompute(img2, None)  # type: ignore

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    # Match descriptors
    matches = bf.match(des1, des2)
    n_inliers = 0
    if len(matches) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)  # type: ignore
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)  # type: ignore
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        n_inliers = np.count_nonzero(mask == 1)

    mr = MatchResult(img1_path,
                     img2_path,
                     len(kp1),
                     len(kp2),
                     n_inliers,
                     (n_inliers / min(len(kp1), len(kp2))))
    return mr


class StitchedImage():
    """
    A class that is used to construct the panorama result / final image.
    """
    def __init__(self, ref_image) -> None:
        pass
class Panorama():

    def __init__(self, path) -> None:
        self.folder_path = path
        self.image_files = get_image_files(path)
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

    def match(self) -> None:
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
        # create Matched_Ids col. 1st match is always the same keypoint, therefore skip
        self.matching_df['Matched_Ids'] = [np.array([m.trainIdx for m in Match[1:]]) for Match in matches]
        self.add_matched_imgids_col()
        # remove (invalid) matches from the same image
        self.matching_df[["Matched_Ids", "Matched_ImgIds"]] = self.matching_df.apply(self.remove_invalid_matches, axis=1)
        return matches

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

    def remove_invalid_matches(self, row) -> pd.Series:
        """
        Remnoves Ids from Matched_Ids that are the are from the same Image (ImgId)

        Args:
            row: a row from a pandas dataframe

        Returns:
            series: a series for containing the updated Matched_Ids and Matched_ImgIds, with invalid values removed.
        """
        matched_ids, img_ids, imgId = row['Matched_Ids'], row['Matched_ImgIds'], row['ImgId'] 
        indeces_to_remove = np.where(img_ids == imgId)
        # create mask to remove values
        mask = np.ones(len(matched_ids), dtype=bool)
        mask[indeces_to_remove] = False
        matched_ids = matched_ids[mask, ...]
        img_ids = img_ids[mask, ...]
        return pd.Series([matched_ids, img_ids])

    def match_stats(self):
        print(f"Match Stats of Images in {self.folder_path}")
        print("{:<12} | {:<12} | {:<9} | {:<9} | {:<16} | {:<9}".format(
            "IMG_1", "IMG_2", "#_FEATS_1",
            "#_FEATS_2", "#_INLIER_MATCHES", "%_INLIERS"))
        print("-" * 80) 
        for i, ref_img in enumerate(self.image_files[:-1]):
            for i, target_img in enumerate(self.image_files[i+1:]):
                mr = match_features(ref_img, target_img)
                print("{:<12} | {:<12} | {:<9} | {:<9} | {:<16} | {:<9.3f}".format(
                    Path(mr.img1).stem,
                    Path(mr.img2).stem,
                    mr.n_feats1,
                    mr.n_feats2,
                    mr.n_inliers,
                    mr.percent_in))



def main(args):
    panorama = Panorama(args.folder_path)
    # print(panorama.image_files)
    panorama.match_stats()
    return


def parse_arguments():
    """
    Parses arguments
    """
    parser = argparse.ArgumentParser(
                    prog='Stitchin App',
                    description='This program stitches images in a folder together to create a panorama',
                    epilog='Have fun!')

    parser.add_argument('folder_path', help="path to folder containing images")  # positional argument
    # parser.add_argument('--log', action='store_true', help="Enables logging to file")  # on/off flag
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    print("Startring Sticthing App")
    args = parse_arguments()
    main(args)
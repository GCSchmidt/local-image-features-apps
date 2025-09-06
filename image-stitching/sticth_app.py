import os
import sys
import re
import argparse
from pathlib import Path
import cv2
import numpy as np

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


class Panorama():

    def __init__(self, path) -> None:
        self.folder_path = path
        self.image_files = get_image_files(path)

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

    def generate_panorama(self):
        pass


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
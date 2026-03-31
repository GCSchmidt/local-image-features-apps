import argparse
import cv2
import numpy as np
import utils.file_utils as file_utils
from image_stitching.models import Panorama, StitchedImage, CVPanorama


def main(args):
    panorama = Panorama(args.folder_path)
    panorama.generate_panorama()
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
    print("Starting Stitching App")
    args = parse_arguments()
    main(args)

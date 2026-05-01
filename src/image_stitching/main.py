import argparse
import logging
import cv2
import numpy as np
import utils.file_utils as file_utils
from core.constants import LOG_DIR 
from image_stitching.pano_maker import PanoMaker


def main(args):
    if args.log:
        logger = logging.getLogger("image_stitching")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        file_handler = logging.FileHandler(LOG_DIR + 'image_stitching.log', mode='w')
        file_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            '%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s\n%(message)s'
        )
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

    file_paths = file_utils.get_image_files(args.folder_path)
    PM = PanoMaker(file_paths)
    PM.generate_panorama()
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
    parser.add_argument('--log', action='store_true', help="Enables logging to file")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    print("Starting Stitching App")
    args = parse_arguments()
    main(args)

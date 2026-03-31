import os
import re
import random as rdm

LOG_PATH = r'/home/schmidtg/coding_projetcs/local-image-features-apps/logs/'
OUTPUT_PATH = r'/home/schmidtg/coding_projetcs/local-image-features-apps/output/'


def get_image_files(path: str) -> list:
    """Returns a list image file pasths inside a dir."""

    image_types = [".jpg", ".png", ".jpeg"]
    image_files = []

    for entry in os.scandir(path):

        if not entry.is_file():
            continue

        filename, extension = os.path.splitext(entry)
        if extension.lower() not in image_types:
            continue

        image_files.append(filename + extension)

    return image_files



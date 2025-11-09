import os
import re


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

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


def get_folders(path: str) -> list:
    """Returns a list of folder paths inside a directory (non-recursive)."""

    folders = []

    for entry in os.scandir(path):
        if entry.is_dir():
            folders.append(entry.path)

    return folders

def rename_files_sequential(folder_path):
    # List all files (ignore directories)
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    # Sort files alphabetically (optional)
    base = int(rdm.random()*1_000_000)
    files.sort()
    for i, filename in enumerate(files, start=1):
        # Get file extension
        ext = os.path.splitext(filename)[1]
        new_name = f"{base+i}_{i}.{ext}"
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)

    

import numpy as np
import os  
import cv2
from utils import file_utils
from core import constants as const

class CVPanorama():

    def __init__(self, path, mode=0):
        self.folder_path = path
        self.image_files = file_utils.get_image_files(path)
        self.images = [cv2.imread(path) for path in self.image_files]
        self.mode = 0  # 0 = cv2.Stitcher_PANORAMA
        if mode == 1:
            # only other valid mode
            self.mode = 1  # 1= cv2.Stitcher_SCANS
        self.output_image: np.ndarray

    def generate_panorama(self) -> None:
        self._pre_processing()
        self.output_image = self._stitch_images()
        self._post_processing()
        output_path = os.path.join(file_utils.OUTPUT_PATH, r"cv_panorama.jpg")

        if self.output_image is None:
            print("Failed to save panorama")
        else:
            print(f"Saving panorama to {output_path}")
            cv2.imwrite(output_path, self.output_image)

    def _stitch_images(self):
        stitcher = cv2.Stitcher_create(self.mode)
        status, panorama = stitcher.stitch(self.images)

        if status == cv2.Stitcher_OK:
            return panorama
        else:
            print("Error during stitching: ", status)
            return None

    def _pre_processing(self):
        self._reduce_res()

    def _reduce_res(self):
        max_width = const.MAX_IMAGE_DIM

        for i, img in enumerate(self.images):
            h, w = img.shape[:2]
            if w > max_width:
                scale = max_width / w
                img = cv2.resize(img, None, fx=scale, fy=scale)     
            # Optional: denoise
            img = cv2.GaussianBlur(img, (3, 3), 0)     
            self.images[i] = img

    def _post_processing(self):
        if self.output_image is None:
            return
        self._crop_convex_hull()

    def _crop_convex_hull(self):
        gray = cv2.cvtColor(self.output_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        biggest = max(cnts, key=cv2.contourArea)
        hull = cv2.convexHull(biggest)

        mask = np.zeros_like(gray)
        cv2.fillConvexPoly(mask, hull, 255)

        x, y, w, h = cv2.boundingRect(hull)
        self.output_image = self.output_image[y:y+h, x:x+w]


if __name__ == "__main__":
    pass
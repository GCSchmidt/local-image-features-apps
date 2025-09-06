import unittest
import os
import time
import random
import sticth_app


class SticthAppTests(unittest.TestCase):

    def test_get_image_files(self):
        test_folder = r'image-stitching/datasets/example-data/flower'
        expected = set([
            f"{test_folder}/1.jpg",
            f"{test_folder}/2.jpg",
            f"{test_folder}/3.jpg",
            f"{test_folder}/4.jpg"
        ])
        actual = set(sticth_app.get_image_files(test_folder))
        self.assertSetEqual(actual, expected)


if __name__ == '__main__':
    print("-"*70, "\nRunning Tests...")
    unittest.main()
# Image Stitching Application
This directory contains the implementation for the **Image Stitching Application** to create panoramas.

## Useful Resources
1. [Potential datasets](https://github.com/visionxiang/Image-Stitching-Dataset?tab=readme-ov-file)
2. [Image Sticthing with openCV](https://pyimagesearch.com/2018/12/17/image-stitching-with-opencv-and-python/)
3. [OpenCV panorama stitching](https://pyimagesearch.com/2016/01/11/opencv-panorama-stitching/)
4. [Real-time panorama and image stitching with OpenCV](https://pyimagesearch.com/2016/01/25/real-time-panorama-and-image-stitching-with-opencv/)
5. [Tips on multiple image stitching](https://stackoverflow.com/questions/24563173/stitch-multiple-images-using-opencv-python) 

## Important Notes 

- **File naming requirement**: Input files must be named with numerical suffixes that specify the stitching order (e.g., "image_001.jpg", "photo_4.png"). This naming convention allows the application to determine the correct sequence for panorama creation without performing complex feature matching between images to establish adjacency. 

- **Future enhancement**: A planned improvement is to make the application more robust by implementing automatic image ordering through feature detection and matching, eliminating the current dependency on sequential file naming.

## Goals / Objectives
1. TO BE DEFINED
2. Speed requirements
3. Accuracy req requirements

## The Dataset
**Source:** The dataset was found at: https://github.com/ppwwyyxx/OpenPano/releases/tag/0.1.

**Details:** It contains 8 sets of images for panorama stitching, and the number of images for each set ranges from 4 to 38.

## TODOs
1. Find a suitable dataset to develop a soultion for
2. Determine Goals/Objective
3. Implement image stitching solution
4. Check if Goals / Objectives are met

## Run the Code

Run with 
```
python3 src/image_stitching/main.py src/image_stitching/datasets/example-data/myself
```
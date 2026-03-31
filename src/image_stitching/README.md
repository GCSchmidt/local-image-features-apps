# Image Stitching Application
This directory contains the implementation for the **Image Stitching Application** to create panoramas.

## Useful Resources
1. [Tips on multiple image stitching](https://stackoverflow.com/questions/24563173/stitch-multiple-images-using-opencv-python) 
2. [Potential datasets](https://github.com/visionxiang/Image-Stitching-Dataset?tab=readme-ov-file)
3. [How to use OpenCV Sticther Class](https://docs.opencv.org/4.x/d8/d19/tutorial_stitcher.html)
4. [OpenCV Image Sticthing Tutorial](https://pyimagesearch.com/2018/12/17/image-stitching-with-opencv-and-python/)
5. [OpenCV panorama stitching Tutorial](https://pyimagesearch.com/2016/01/11/opencv-panorama-stitching/)
6. [OpenCV Real-time image stitching Tutorial](https://pyimagesearch.com/2016/01/25/real-time-panorama-and-image-stitching-with-opencv/)


## Important Notes 

## Goals / Objectives
Get my implementation to produce the same result as Opencv's implementation in less than double the time. 

## The Dataset
**Source:** The dataset was found at: https://github.com/ppwwyyxx/OpenPano/releases/tag/0.1.

**Details:** It contains 8 sets of images for panorama stitching, and the number of images for each set ranges from 4 to 38.

## TODOs
1. Remove the use of pandas
2. Get my own implemantations to work

## Run the Code

Run with 
```
python3 src/image_stitching/main.py src/image_stitching/datasets/example-data/myself
```
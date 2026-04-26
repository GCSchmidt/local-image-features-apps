# constants

# Panorama
KNN_FOR_PANORAMA = 4  # number best macthes to find for each feature 
M_CANDIDATE_IMAGES = 6  # number of images to check for connections
INLIER_THRESHOLD = 0.25  # Percentage of inliers in the overlapping area required to confirm connection between images
GOOD_MATCH_RATIO = 0.75  # ratio needed for best match to be better than 2nd
MIN_INLIERS = 30

# File Paths
LOG_DIR = r'/home/schmidtg/coding_projetcs/local-image-features-apps/logs/'
OUTPUT_DIR = r'/home/schmidtg/coding_projetcs/local-image-features-apps/output/'
import os
from utils.utils import*
from tqdm import tqdm
from calibration.calibdata import getProjectionMatrices
from scipy.spatial.transform import Rotation as R

## Loading images

images_L = []
images_R = []
images_view_L = []
images_view_R = []
L_folder = 'src\stereo_video\L'
R_folder = 'src\stereo_video\R'
n_immagini = 20
frame_skip = 1
startFrame = 10

# Get a sorted list of file names in the left and right folders
left_files = sorted(os.listdir(L_folder))
right_files = sorted(os.listdir(R_folder))

# Iterate through the sorted file names
for i in tqdm(range(1, n_immagini + 1)):
    # Use the sorted index to load the images in the correct order
    L_file_path = os.path.join(L_folder, left_files[i])
    R_file_path = os.path.join(R_folder, right_files[i])
    images_L.append(load_image(L_file_path))
    images_R.append(load_image(R_file_path))

## Defining processing tools

extractor = cv2.SIFT_create()
matcher = cv2.BFMatcher()

## Matching points

# Match pivot sterio pair
 
imgL, imgR = getStereoPair(0, images_L, images_R)

pL, pR, f = matchPair(imgL, imgR, extractor, matcher)
displayMatchedPair(imgL, imgR, pL, pR)
for i in tqdm(range(1, n_immagini)):
    imgL, imgR = getStereoPair(i, images_L, images_R)
    if(len(pL)<200):
        pL, pR, f = matchPair(imgL, imgR, extractor, matcher)
    else:
        pL, pR, f = matchPairFilter(imgL, imgR, extractor, matcher, f)
        # save_points_to_csv("stereo_points_"+ str(i) +".csv", pL, pR)
    displayMatchedPair(imgL, imgR, pL, pR)
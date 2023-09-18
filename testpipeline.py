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
# displayMatchedPair(imgL, imgR, pL, pR)
# for i in tqdm(range(1, n_immagini)):
#     imgL, imgR = getStereoPair(i, images_L, images_R)
#     if(len(pL)<200):
#         pL, pR, f = matchPair(imgL, imgR, extractor, matcher)
#     else:
#         pL, pR, f = matchPairFilter(imgL, imgR, extractor, matcher, f)
#         # save_points_to_csv("stereo_points_"+ str(i) +".csv", pL, pR)
#     displayMatchedPair(imgL, imgR, pL, pR)

## Stereo triangulation
P1, P2 = getProjectionMatrices("D:\PrismaLab\Progetti\OpticalBox\VO\calibration\calibration_data.json")

p3ds = []
for uv1, uv2 in zip(pL, pR):
    _p3d = DLT(P1, P2, uv1, uv2)
    p3ds.append(_p3d)

p3ds = np.array(p3ds)

# Create a rotation matrix from Euler angles
# r = R.from_euler('xyz', [-43, 0, 0], degrees=True)
# rotation_matrix = r.as_matrix()
# p3ds = np.dot(rotation_matrix, p3ds.T).T

max_distance = 1.0e3

point_distances = np.linalg.norm(p3ds, axis=1)

# Filter out points that are too far and have a negative Y coordinate
# filtered_p3ds = []
# for i, p3d in enumerate(p3ds):
#     if point_distances[i] <= max_distance and p3d[2] >= 0:
#         filtered_p3ds.append(p3d)

# # Convert the filtered points to a NumPy array
# filtered_p3ds = np.array(filtered_p3ds)

# p3ds = filtered_p3ds

# Scatter plot the 3D points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
p3ds = np.array(p3ds)
ax.scatter(p3ds[:, 0], p3ds[:, 1], p3ds[:, 2], c='b', marker='o')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
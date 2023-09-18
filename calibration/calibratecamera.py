import cv2 as cv
import numpy as np
import os

def calibrate_camera(images_folder):
    images_names = sorted(os.listdir(images_folder))
    images = []
    for imname in images_names:
        im = cv.imread(os.path.join(images_folder, imname), 1)
        images.append(im)

    # Criteria used by the checkerboard pattern detector.
    # Change this if the code can't find the checkerboard.
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    rows = 6  # Number of checkerboard rows.
    columns = 8  # Number of checkerboard columns.
    world_scaling = 1.0  # Change this to the real world square size. Or not.

    # Coordinates of squares in the checkerboard world space.
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp

    # Frame dimensions. Frames should be the same size.
    width = images[0].shape[1]
    height = images[0].shape[0]

    # Pixel coordinates of checkerboards.
    imgpoints = []  # 2D points in image plane.

    # Coordinates of the checkerboard in checkerboard world space.
    objpoints = []  # 3D points in real world space.

    for frame_idx, frame in enumerate(images):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Find the checkerboard.
        ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)

        if ret:
            print(f"Checkerboard found in image {frame_idx}")
            # Convolution size used to improve corner detection. Don't make this too large.
            conv_size = (11, 11)

            # OpenCV can attempt to improve the checkerboard coordinates.
            corners = cv.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)

            objpoints.append(objp)
            imgpoints.append(corners)
        else:
            print(f"Checkerboard not found in image {frame_idx}")

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
    print('rmse:', ret)
    print('camera matrix:\n', mtx)
    print('distortion coefficients:', dist)
    print('Rs:\n', rvecs)
    print('Ts:\n', tvecs)

    return mtx, dist

def stereo_calibrate(mtx1, dist1, mtx2, dist2, left_frame_folder, right_frame_folder):
    left_files = sorted(os.listdir(left_frame_folder))
    right_files = sorted(os.listdir(right_frame_folder))
    c1_images = []
    c2_images = []

    if len(left_files) != len(right_files):
        print("Error: Mismatch in the number of left and right images.")
        return None, None

    for i in range(len(left_files)):
        left_im = cv.imread(os.path.join(left_frame_folder, left_files[i]), 1)
        right_im = cv.imread(os.path.join(right_frame_folder, right_files[i]), 1)
        c1_images.append(left_im)
        c2_images.append(right_im)

    print(f"Number of left images: {len(c1_images)}")
    print(f"Number of right images: {len(c2_images)}")

    # Change this if stereo calibration is not good.
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

    rows = 6  # Number of checkerboard rows.
    columns = 8  # Number of checkerboard columns.
    world_scaling = 1.0  # Change this to the real world square size. Or not.

    # Coordinates of squares in the checkerboard world space.
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp

    # Frame dimensions. Frames should be the same size.
    width = c1_images[0].shape[1]
    height = c1_images[0].shape[0]

    # Pixel coordinates of checkerboards for left and right images.
    imgpoints_left = []  # 2D points in the image plane for left camera.
    imgpoints_right = []  # 2D points in the image plane for right camera.

    # Coordinates of the checkerboard in checkerboard world space.
    objpoints = []  # 3D points in real world space.

    for frame_idx, (frame1, frame2) in enumerate(zip(c1_images, c2_images)):
        gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv.findChessboardCorners(gray1, (6, 8), None)
        c_ret2, corners2 = cv.findChessboardCorners(gray2, (6, 8), None)

        if c_ret1 and c_ret2:
            print(f"Checkerboard found in stereo pair {frame_idx}")
            corners1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

            cv.drawChessboardCorners(frame1, (6, 8), corners1, c_ret1)
            cv.imshow('Left Image with Corners', frame1)
            cv.drawChessboardCorners(frame2, (6, 8), corners2, c_ret2)
            cv.imshow('Right Image with Corners', frame2)
            k = cv.waitKey(500)

            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)
        else:
            print(f"Checkerboard not found in stereo pair {frame_idx}")

    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right, mtx1, dist1, mtx2, dist2, (width, height), criteria=criteria, flags=stereocalibration_flags
    )

    print('Stereo calibration result:', ret)
    return R, T

# Update image folder paths as needed.
left_images_folder = 'D:\PrismaLab\Progetti\OpticalBox\VO\calibration\calibration_images\L'
right_images_folder = 'D:\PrismaLab\Progetti\OpticalBox\VO\calibration\calibration_images\R'

mtx1, dist1 = calibrate_camera(images_folder=left_images_folder)
mtx2, dist2 = calibrate_camera(images_folder=right_images_folder)

if mtx1 is not None and mtx2 is not None:
    R, T = stereo_calibrate(mtx1, dist1, mtx2, dist2, left_images_folder, right_images_folder)
    if R is not None and T is not None:
        print('Rotation matrix R:\n', R)
        print('Translation vector T:\n', T)
    else:
        print('Stereo calibration failed.')
else:
    print('Camera calibration failed.')


import json

# Define a dictionary to store the calibration data
calibration_data = {
    "mtx1": mtx1.tolist(),
    "dist1": dist1.tolist(),
    "mtx2": mtx2.tolist(),
    "dist2": dist2.tolist(),
    "R": R.tolist(),
    "T": T.tolist()
}

# Specify the path to the JSON file where you want to save the data
json_file_path = "calibration_data.json"

# Save the calibration data to the JSON file
with open(json_file_path, "w") as json_file:
    json.dump(calibration_data, json_file)

print(f"Calibration data saved to {json_file_path}")



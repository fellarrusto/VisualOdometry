import json
import numpy as np


def getCalibrationData(file_name):
    # Specify the path to the JSON file containing the calibration data
    json_file_path = file_name

    # Load the calibration data from the JSON file
    with open(json_file_path, "r") as json_file:
        calibration_data = json.load(json_file)

    # Convert the loaded data back to NumPy arrays
    mtx1 = np.array(calibration_data["mtx1"])
    dist1 = np.array(calibration_data["dist1"])
    mtx2 = np.array(calibration_data["mtx2"])
    dist2 = np.array(calibration_data["dist2"])
    R = np.array(calibration_data["R"])
    T = np.array(calibration_data["T"])

    return mtx1, dist1, mtx2, dist2, R, T

def getProjectionMatrices(file_name):
    # Specify the path to the JSON file containing the calibration data
    json_file_path = file_name

    # Load the calibration data from the JSON file
    with open(json_file_path, "r") as json_file:
        calibration_data = json.load(json_file)

    # Convert the loaded data back to NumPy arrays
    mtx1 = np.array(calibration_data["mtx1"])
    dist1 = np.array(calibration_data["dist1"])
    mtx2 = np.array(calibration_data["mtx2"])
    dist2 = np.array(calibration_data["dist2"])
    R = np.array(calibration_data["R"])
    T = np.array(calibration_data["T"])

    #RT matrix for C1 is identity.
    RT1 = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
    P1 = mtx1 @ RT1 #projection matrix for C1
    
    #RT matrix for C2 is the R and T obtained from stereo calibration.
    RT2 = np.concatenate([R, T], axis = -1)
    P2 = mtx2 @ RT2 #projection matrix for C2

    return P1, P2
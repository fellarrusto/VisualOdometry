from matplotlib import pyplot as plt
import numpy as np
import cv2
import csv
from scipy import linalg


def filterFeatures(k, d, mtc):
   idxs = [m.queryIdx for m in mtc]
   k = np.array(k)
   d = np.array(d)
   k = k[idxs]
   d = d[idxs]
   return k, d

def load_image(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def getStereoPair(i, images_L, images_R):
    return (images_L[i], images_R[i])

def getGoodFeatures(matches):
    mtc = []
    for m,n in matches:
        if m.distance < 0.8*n.distance:
            mtc.append(m)
    return mtc

def matchPair(imgL, imgR, extractor, matcher):
    kpL, desL = extractor.detectAndCompute(imgL, None)
    kpR, desR = extractor.detectAndCompute(imgR, None)
    matches = getGoodFeatures(matcher.knnMatch(desL,desR,k=2))

    pointsL = [kpL[m.queryIdx].pt for m in matches]
    pointsR = [kpR[m.trainIdx].pt for m in matches]

    k, d = filterFeatures(kpL, desL, matches)
    return pointsL, pointsR, d


def matchPairFilter(imgL, imgR, extractor, matcher, f):
    kpL, desL = extractor.detectAndCompute(imgL, None)
    kpR, desR = extractor.detectAndCompute(imgR, None)

    # Select persistent features in frame L
    m1 = getGoodFeatures( matcher.knnMatch(desL,f,k=2))
    kpL, desL = filterFeatures(kpL, desL, m1)

    matches = getGoodFeatures(matcher.knnMatch(desL,desR,k=2))

    # Get matched keypoints' coordinates in both images
    pointsL = [kpL[m.queryIdx].pt for m in matches]
    pointsR = [kpR[m.trainIdx].pt for m in matches]

    k, d = filterFeatures(kpL, desL, matches)
    return pointsL, pointsR, d

def DLT(P1, P2, point1, point2):
 
    A = [point1[1]*P1[2,:] - P1[1,:],
         P1[0,:] - point1[0]*P1[2,:],
         point2[1]*P2[2,:] - P2[1,:],
         P2[0,:] - point2[0]*P2[2,:]
        ]
    A = np.array(A).reshape((4,4)) 
    B = A.transpose() @ A
    
    U, s, Vh = linalg.svd(B, full_matrices = False)
    return Vh[3,0:3]/Vh[3,3]


def displayMatchedPair(imgL, imgR, pL, pR):
    pL = np.array(pL)
    pR = np.array(pR)
    # Display the images with matched keypoints
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.imshow(imgL, cmap='gray')
    ax1.scatter(pL[:, 0], pL[:, 1], c='r', marker='o', s=10)
    ax1.set_title('Image 1 with Keypoints')

    ax2.imshow(imgR, cmap='gray')
    ax2.scatter(pR[:, 0], pR[:, 1], c='b', marker='o', s=10)
    ax2.set_title('Image 2 with Keypoints')

    plt.tight_layout()
    plt.show()

def save_points_to_csv(filename, points_L, points_R):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["xL", "yL", "xR", "yR"])  # Write header row
        for point in points_L:
            xL, yL = point
            xR, yR = points_R[points_L.index(point)]  # Match the corresponding point from points_R
            writer.writerow([xL, yL, xR, yR])
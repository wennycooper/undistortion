import numpy as np
import cv2
import glob
import codecs, json
import yaml

# images = glob.glob('chessboard_*')
images = glob.glob('left*.png')


# load calibration parameters
file_path = "ost.yaml"
with open(file_path, 'r') as stream:
    f = yaml.load(stream)

mtx = np.array(f["camera_matrix"]['data']).reshape((3,3))
dist = np.array(f["distortion_coefficients"]["data"])

# Undistortion
for fname in images:
    img = cv2.imread(fname)
    h, w = img.shape[:2]       # (1080, 1920)

    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    cv2.imwrite(fname + "_result.jpg", dst)

    cv2.imshow("img", dst)
    cv2.waitKey(200)

# Re-projection Error
#mean_error = 0
#for i in xrange(len(objpoints)):
#    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
#    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
#    mean_error += error
#
#print "mean error: ", mean_error/len(objpoints)


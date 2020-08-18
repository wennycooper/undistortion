import numpy as np
import cv2
#import glob
#import codecs, json
import yaml
from optparse import OptionParser

usage = "usage: %prog -y YAML_FILENAME -i INPUT_FILENAME -o OUTPUT_FILENAME"

parser = OptionParser(usage)

parser.add_option("-y", "--yaml", dest="yaml_filename", help="YAML FILENAME")
parser.add_option("-i", "--i", dest="input_filename", help="INPUT FILENAME")
parser.add_option("-o", "--o", dest="output_filename", help="OUTPUT FILENAME")

(options, args) = parser.parse_args()

if not options.yaml_filename:
    parser.error("yaml_filename not given")

if not options.input_filename:
    parser.error("input_filename not given")

if not options.output_filename:
    parser.error("output_filename not given")

yaml_filename = options.yaml_filename
input_filename = options.input_filename
output_filename = options.output_filename

cap = cv2.VideoCapture(input_filename)

# Find OpenCV version
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

if int(major_ver)  < 3 :
    fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
else:
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

# images = glob.glob('chessboard_*')
# images = glob.glob('left*.png')


# load calibration parameters
#file_path = "ost.yaml"

with open(yaml_filename, 'r') as stream:
    f = yaml.load(stream)

mtx = np.array(f["camera_matrix"]['data']).reshape((3,3))
dist = np.array(f["distortion_coefficients"]["data"])

width = int(cap.get(3))
height = int(cap.get(4))

fourcc = cv2.VideoWriter_fourcc('D','I','V','X')

out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

# Undistortion
while (cap.isOpened()):
    #img = cv2.imread(fname)
    ret, img = cap.read()
    if ret != True:
        break

    h, w = img.shape[:2]       # (1080, 1920)

    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    out.write(dst)

    cv2.imshow("img", dst)
    cv2.waitKey(1)

cap.release()
out.release()


import numpy as np
import cv2

from veinWork.utils.cv_functions import *


def nothing(x):
    pass

# Create a black image, a window
img = cv2.imread("F:/2projectInfo/blood_robot/original/needle4.png")
cv2.imshow("image", img)
cv2.namedWindow('image')
# create trackbars for color change
# cv2.createTrackbar('niblack_sidelength ', 'image', 30, 50, nothing)  #49
# cv2.createTrackbar('niblack_threk', 'image', 0, 1, nothing)  #0.1
# cv2.createTrackbar('winsize ', 'image', 3, 7, nothing)  #5
cv2.createTrackbar('b_openside ', 'image', 0, 255, nothing)  #1
cv2.createTrackbar('b_closedside', 'image', 0, 255, nothing)  #7
cv2.createTrackbar('b_blurside ', 'image', 0, 255, nothing)  #5
# cv2.createTrackbar('erodeside ', 'image', 0, 255, nothing)  #5
# cv2.createTrackbar('dilateside1', 'image', 0, 255, nothing)  #9
# cv2.createTrackbar('dilateside1 ', 'image', 0, 255, nothing)  #5

while(1):

    # get current positions of four trackbars
    # sidelength = cv2.getTrackbarPos('niblack_sidelength', 'image')
    # threk = cv2.getTrackbarPos('niblack_threk', 'image')
    # winsize = cv2.getTrackbarPos('winsize', 'image')
    b_openside = cv2.getTrackbarPos('b_openside', 'image')
    b_closedside = cv2.getTrackbarPos('b_closedside', 'image')
    b_blurside = cv2.getTrackbarPos('b_blurside', 'image')
    # erodeside = cv2.getTrackbarPos('erodeside', 'image')
    # dilateside1 = cv2.getTrackbarPos('dilateside1', 'image')
    # dilateside2 = cv2.getTrackbarPos('dilateside2', 'image')

    # s = cv.getTrackbarPos(switch, 'image')
    src = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    seg_img1 = thre_segment(src, 67, 0.1)
    # seg_blur_img = cv2.medianBlur(seg_img, 5)
    # cv2.imshow("seg_blur_img", seg_blur_img)
    denoise_img = blood_denoise(seg_img1, b_openside, b_closedside, b_blurside)
    # erode_img = erode(denoise_img, erodeside, dilateside1, dilateside2)
    # fill_blood_img = get_contours(erode_img, 2000)
    # cv2.imshow("seg_img", seg_img)
    if cv2.waitKey(10) == ord('q'):
        break
cv2.destroyAllWindows()
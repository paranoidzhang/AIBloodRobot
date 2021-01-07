from veinWork.utils.cv_functions import *

src = cv2.imread("F:/2projectInfo/blood_robot/original/needle4.png")

# src = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# seg_img = thre_segment(src, 67, 0.1)
# # cv2.imshow("seg_img1", seg_img)
# denoise_img2 = blood_denoise(seg_img, 3, 7, 5) # b_openside, b_closedside, b_blurside
# # cv2.imshow("denoise_img2", denoise_img2)
# # seg_blur_img1 = cv2.medianBlur(seg_img, 3)
# # cv2.imshow("seg_blur_img1", seg_blur_img1)
# clear_img = blood_clear_background(denoise_img2)
# # cv2.imshow("clear_img", clear_img)
# fill_blood_img = get_contours(clear_img,2000)
# # cv2.imshow("fill_blood_img", fill_blood_img)
# # show_img(fill_blood_img)
needle_img = get_needle2(src)
blur_img = needle_clear_background(needle_img)
denoise_img1 = needle_denoise(blur_img, 5, 3, 5)
cv2.imshow("denoise_img1",denoise_img1)
fill_needle_img = get_contours(denoise_img1, 2000)
cv2.imshow("fill_needle_img", fill_needle_img)
# denoise_img2 = needle_denoise(blur_img, 3, 7, 3)
# cv2.imshow("denoise_img2",denoise_img2)

cv2.waitKey(0)
cv2.destroyAllWindows()
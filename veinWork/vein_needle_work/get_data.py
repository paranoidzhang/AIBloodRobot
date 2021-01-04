from veinWork.camera.nir_camara import init_nir_img, get_nir_img
from veinWork.utils.cv_functions import *


def get_needle(src):
    while True:
        src = src[62:420, :]
        img = src.copy()
        # step 1：获取针的二值图
        needle_img = get_needle(src)

        # step 2：漫水填充去除边界干扰
        blur_img = needle_clear_background(needle_img)

        # step 3：图片去噪+细化采血针轮廓
        # hough_img = get_needle_line(blur_img,img)

        # 图片去噪
        denoise_img = needle_denoise(blur_img, n_openside, n_closedside, n_blurside)

        # 细化采血针轮廓
        thin_needle_img = thin_needle(denoise_img)

        needle_line = get_line(thin_needle_img)
        if needle_line:
            needle_theta = get_theta(needle_line)
            break

    # step 4：提取针头坐标
    row, col = get_needle_point(thin_needle_img)

    # step 5：提取ROI区域
    blood_roi = src[row:src.shape[0], col - 150:col + 150]
    # blood_roi = src[row:src.shape[0], 0:src.shape[1]]
    return blood_roi, needle_theta


def get_vein(src):
    # step 6：识别血管及方向
    # step6-1:获取肘部区域
    veinimg = get_vein(src)
    timg = cv2.cvtColor(veinimg, cv2.COLOR_BGR2GRAY)

    # step6-2:局部阈值分割
    seg_img = thre_segment(timg, sidelength, threk)

    # step6-3:中值滤波去噪
    seg_blur_img = cv2.medianBlur(seg_img, winsize)

    # step6-4：开运算闭运算去噪
    # denoise_img = blood_denoise(seg_blur_img, b_openside, b_closedside, b_blurside)

    # step6-5：清除背景操作
    clear_img = blood_clear_background(seg_blur_img)

    # step6-6：腐蚀膨胀操作
    erode_img = erode(clear_img, erodeside, dilateside1, dilateside2)

    # step6-7：填充孔洞操作
    fill_img = get_contours(erode_img)

    # step 7：细化血管 提取血管骨架
    thin_blood_img = thin_needle(fill_img)

    blood_line = get_line(thin_blood_img)
    blood_theta = get_theta(blood_line)
    return blood_theta


def blood_needle_main(src):
    blood_roi, needle_theta = get_needle(src)
    blood_theta = get_vein(blood_roi)
    # step 8:比较针头与血管方向
    res = cmp_theta(blood_theta, needle_theta)
    return res


def judge_theta():
    cap = init_nir_img()
    count = 0
    for i in range(10):
        src = get_nir_img(cap)
        res = blood_needle_main(src)
        if res:
            count = count + 1
    if count > 7:
        return True
    else:
        return False


if __name__ == "__main__":
    res = judge_theta()
    if res:
        print("success")
    else:
        # 结束采血流程
        pass

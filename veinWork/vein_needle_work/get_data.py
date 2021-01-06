from veinWork.camera.nir_camara import init_nir_img, get_nir_img
from veinWork.utils.cv_functions import *
from veinWork.utils.config_util import Config

config = Config()
sidelength = config.get_thresh_params2_config()['sidelength']
threk = config.get_thresh_params2_config()['threk']
b_openside = config.get_thresh_params2_config()['b_openside']
b_closedside = config.get_thresh_params2_config()['b_closedside']
b_blurside = config.get_thresh_params2_config()['b_blurside']
gamma = config.get_thresh_params2_config()['gamma']
winsize = config.get_thresh_params2_config()['winsize']
blurside = config.get_thresh_params2_config()['blurside']
threfactor = config.get_thresh_params2_config()['threfactor']
erodeside = config.get_thresh_params2_config()['erodeside']
dilateside1 = config.get_thresh_params2_config()['dilateside1']
dilateside2 = config.get_thresh_params2_config()['dilateside2']
n_openside = config.get_thresh_params2_config()['n_openside']
n_closedside = config.get_thresh_params2_config()['n_closedside']
n_blurside = config.get_thresh_params2_config()['n_blurside']

def get_needle_img(src):
    src = src[62:420, :]
    img = src.copy()
    # step 1：获取针的二值图
    needle_img = get_needle2(src)
    # cv2.imshow("needle_img",needle_img)

    # step 2：漫水填充去除边界干扰
    blur_img = needle_clear_background(needle_img)
    # cv2.imshow("blur_img", blur_img)

    # step 3：图片去噪+细化采血针轮廓
    # hough_img = get_needle_line(blur_img,img)
    # 图片去噪
    denoise_img = needle_denoise(blur_img, n_openside, n_closedside, n_blurside)
    cv2.imshow("denoise_img", denoise_img)

    fill_needle_img = get_contours(denoise_img, 500)
    cv2.imshow("fill_needle_img", fill_needle_img)


    # 细化采血针轮廓
    thin_needle_img = thin_needle(fill_needle_img)
    # cv2.imshow("thin_needle_img", thin_needle_img)

    needle_line = get_line(thin_needle_img)
    # (3,348),(54,351)
    if needle_line:
        needle_theta = get_theta(needle_line)
        # theta = 3.3664606634298013;2.2457425658950716;2.33730585912382;2.3859440303888126;2.602562202499806


    # step 4：提取针头坐标
    row, col = get_needle_point(thin_needle_img)  # 54,351 ; 53,371;57,372;56,372;56,372;56,373
    print(row, col)
    cv2.circle(src, (col,row), 1, (0, 255, 0), 4)
    cv2.imshow('circlesrc', src)

    # step 5：提取ROI区域
    # blood_roi = src[row:src.shape[0], 0:src.shape[1]]
    blood_roi = src[row:src.shape[0], col - 150:col + 150]  # 157 122 ;194 122;166 95
    cv2.imshow("blood_roi", blood_roi)
    # blood_roi = src[row:src.shape[0], 0:src.s hape[1]]
    return blood_roi, needle_theta, src, row, col
    # return needle_theta


def get_vein_img(src):
    # step 6：识别血管及方向
    # step6-1:获取肘部区域
    # veinimg = get_vein(src)
    # img = src.copy()
    timg = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # step6-2:局部阈值分割
    seg_img = thre_segment(timg, sidelength, threk)
    cv2.imshow("seg_img", seg_img)

    # step6-3:中值滤波去噪
    seg_blur_img = cv2.medianBlur(seg_img, winsize)

    # step6-4：开运算闭运算去噪
    denoise_img = blood_denoise(seg_blur_img, b_openside, b_closedside, b_blurside)

    # step6-5：清除背景操作
    clear_img = blood_clear_background(denoise_img)
    # cv2.imshow("clear_img", clear_img)

    # step6-6：腐蚀膨胀操作
    erode_img = erode(clear_img, erodeside, dilateside1, dilateside2)

    # step6-7：填充孔洞操作
    fill_blood_img = get_contours(erode_img,2000)
    cv2.imshow("fill_blood_img", fill_blood_img)

    # step 7：细化血管 提取血管骨架

    thin_blood_img = thin_needle(fill_blood_img)
    cv2.imshow("thin_blood_img", thin_blood_img)

    # hough_img = get_needle_line(thin_blood_img, thin_blood_img)
    # cv2.imshow("hough_img", hough_img)


    blood_line = get_line(thin_blood_img)
    cv2.line(src, (blood_line[0][1], blood_line[0][0]), (blood_line[1][1], blood_line[1][0]), (0, 255, 0), 1)
    cv2.imshow('linesrc', src)
    blood_theta = get_theta(blood_line)
    return thin_blood_img,blood_theta


def blood_needle_main(src):
    blood_roi, needle_theta = get_needle_img(src)
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
    # res = judge_theta()
    # if res:
    #     print("success")
    # else:
    #     # 结束采血流程
    #     pass
    cap = init_nir_img()
    while 1:
        src = get_nir_img(cap)
        # cv2.imshow("src", src)
        blood_roi, needle_theta, src, row, col = get_needle_img(src)
        # cv2.waitKey(0)
        # 方法一
        thin_blood_img,blood_theta = get_vein_img(blood_roi)
        cv2.imshow("blood_img",thin_blood_img)

        res = cmp_theta(blood_theta, needle_theta)
        print(needle_theta,blood_theta,res)

        if cv2.waitKey(100) & 0xff == ord('q'):
            break
    cv2.destroyAllWindows()



import cv2
import numpy as np
from numpy import *
from skimage.filters import threshold_niblack
from scipy.signal import *
from scipy.sparse import *
from skimage.morphology import thin
from PIL.Image import open as ImOpen,fromarray,BILINEAR
import matplotlib.pyplot as plt


def gamma_transfer(img, gamma):
    """伽马变换
    :param img: 输入图像
    :param gamma: gamma阈值
    :return: 输出图像
    """
    fi = img / 255.0
    # gamma = 1.5
    out = np.power(fi, gamma)
    return out


def thre_segment(img, sidelength, threk):
    """局部阈值分割
    :param img:输入图像
    :param sidelength:Niblack模板边长
    :param threk:补偿权值
    :return:输出图像
    """
    thresh_niblack = threshold_niblack(img, window_size=sidelength, k=threk)
    binary_niblack = img > thresh_niblack
    seg_img = binary_niblack.astype(int) * 255
    res = np.uint8(seg_img)
    return res


def blood_denoise(img, b_openside, b_closedside, b_blurside):
    """开闭运算图像去噪（血管）
    :param img: 输入图像
    :param b_openside: 开运算模板边长
    :param b_closedside: 闭运算模板边长
    :param b_blurside: 中值滤波模板边长
    :return: 输出图像
    """
    # 考虑添加顶帽变换（校正不均匀光照 得到原图中灰度较亮的区域）、底帽变换（得到原图中灰度较暗的区域）
    bgr_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # im = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # openkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (b_openside, b_openside))
    # # 开运算 先腐蚀后膨胀 消除暗背景下的较亮区域、平滑边界、纤细处分离物体
    # opening = cv2.morphologyEx(bgr_img, cv2.MORPH_OPEN, openkernel)
    closedkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (b_closedside, b_closedside))
    # 闭运算 先膨胀后腐蚀 填充白色区域里细小黑色空洞的区域 连接临近物体
    closed = cv2.morphologyEx(bgr_img, cv2.MORPH_CLOSE, closedkernel)
    result = cv2.medianBlur(closed, b_blurside)
    denoise_img = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    return denoise_img


def medianBlur(image, winsize):
    """中值滤波
    :param image: 输入图像
    :param winsize: 中值滤波参数
    :return: 输出图像
    """
    # 图像的高，宽
    info = image.shape
    rows = info[0]
    cols = info[1]
    # rows, cols, channel

    # 窗口的宽高均为奇数
    winH, winW = winsize
    halfWinH = int((winH - 1) / 2)
    halfWinW = int((winW - 1) / 2)

    # 中值滤波后的输出图像
    medianBlurImage = np.zeros(image.shape, image.dtype)
    for r in range(rows):
        for c in range(cols):
            # 判断边界
            rTop = 0 if r - halfWinH < 0 else r - halfWinH
            rBottom = rows - 1 if r + halfWinH > rows - 1 else r + halfWinH
            cLeft = 0 if c - halfWinW < 0 else c - halfWinW
            cRight = cols if c + halfWinW > cols - 1 else c + halfWinW

            # 取邻域
            region = image[rTop:rBottom + 1, cLeft:cRight + 1]
            # 求中值
            medianBlurImage[r][c] = np.median(region)
    return medianBlurImage


def blood_clear_background(src):
    """漫水填充算法（血管）
    :param src: 输入图像
    :return: 输出图像
    """
    # img = src[0:src.shape[0], 0:src.shape[1]]  # [高 ，宽]
    # blur_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 灰度图
    height, width = src.shape  # 获取图片宽高
    # 去除黑色背景，seedPoint代表初始种子，进行四次，即对四个角都做一次，可去除最外围的黑边
    blur_img = cv2.floodFill(src, mask=None, seedPoint=(width - 1, 1), newVal=(255, 255, 255))[1]  # 从右侧漫水填充
    blur_img = cv2.floodFill(src, mask=None, seedPoint=(2,int(height/2)), newVal=(255, 255, 255))[1] #从左侧漫水填充
    blur_img = cv2.floodFill(src, mask=None, seedPoint=(0, height - 1), newVal=(255, 255, 255))[1]  #从左下角漫水填充
    blur_img = cv2.floodFill(src, mask=None, seedPoint=(width - 1, height - 1), newVal=(255, 255, 255))[1] #从右下角漫水填充
    return src



def direct_valley(img, threfactor):
    """基于方向的谷型检测
    :param img: 输入图像
    :param threfactor: 灰度放大系数
    :return:
    """
    pix = array(img)
    row = array([0, 2, 4, 6, 8])
    value = array([3, -1, -4, -1, 3])
    A = {}
    for i in range(4):
        if i == 0:
            col = row
        elif i == 1:
            col = array([2, 3, 4, 5, 6])
        elif i == 2:
            col = array([4, 4, 4, 4, 4])
        else:
            col = array([6, 5, 4, 3, 2])
        A[i] = csc_matrix((value, (row, col)), shape=(9, 9)).toarray()
        A[i + 4] = rot90(A[i])

    pixdict = {}
    for i in range(8):
        pixdict[i] = convolve2d(pix, A[i], 'same')

    m, n = pix.shape
    for x in range(m):
        for y in range(n):
            valmax = -1000
            for i in range(8):
                if pixdict[i][x, y] >= valmax:
                    valmax = pixdict[i][x, y]
            if valmax >= 0:
                pix[x, y] = valmax
            else:
                pix[x, y] = 0

    nozero_num = count_nonzero(pix)
    nozero_avg = sum(pix) / nozero_num

    for x in range(m):
        for y in range(n):
            if pix[x, y] >= (threfactor * nozero_avg):
                pix[x, y] = threfactor * nozero_avg
    return pix


def erode(img,erodeside,dilateside1,dilateside2):
    """腐蚀操作
    :param img: 输入图像
    :param erodeside: 腐蚀参数
    :param dilateside1: 膨胀参数1
    :param dilateside2: 膨胀参数2
    :return: 输出图像
    """
    # bgr_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    erodekernel = cv2.getStructuringElement(cv2.MORPH_RECT, (erodeside, erodeside))
    eroded = cv2.erode(img, erodekernel)    #扩大黑色区域
    dilatekernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (dilateside1, dilateside1))
    dilated = cv2.dilate(eroded, dilatekernel1)  # 扩大白色区域
    # erode_img = cv2.cvtColor(dilated, cv2.COLOR_BGR2GRAY)
    return dilated


def dilate(img,dilateside):
    """膨胀操作
    :param img:
    :param dilateside: 膨胀参数
    :return:
    """
    bgr_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    dilatekernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilateside, dilateside))
    dilated = cv2.dilate(bgr_img,dilatekernel)
    dilate_img = cv2.cvtColor(dilated, cv2.COLOR_BGR2GRAY)
    return dilate


def get_vein(src):
    """获取血管区域
    :param src: 输入图像
    :return: 输出图像
    """
    mask = cv2.inRange(src, (0, 0, 0), (180, 255, 46))  # 分离出黑色区域
    mask1 = cv2.bitwise_not(mask)
    # 拿到肘部图像
    elbow = cv2.bitwise_and(src, src, mask=mask1)
    # step 1:创建一个与原图一致的白色背景图
    background = np.zeros(src.shape, src.dtype)
    background[:, :, :] = 0
    # step 2:创建mask1的剩余区域为mask2 将mask1设置为白色
    mask2 = cv2.bitwise_not(mask1)
    dst = cv2.bitwise_or(elbow, background, mask=mask2)
    # step 3：显示肘部图像（并将多余图像显示为白色）
    elbowImg = cv2.add(elbow, dst)
    return elbowImg


def get_contours(img,size):
    """填充细小孔洞区域（颜色转换为白色）
    :param img:输入图像
    :return:输出图像
    """
    contours= cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    n = len(contours)  # 轮廓的个数
    cv_contours = []
    for contour in contours[1]:
        area = cv2.contourArea(contour)
        if area <= size:
            cv_contours.append(contour)
            # x, y, w, h = cv2.boundingRect(contour)
            # img[y:y + h, x:x + w] = 255
        else:
            continue

    cv2.fillPoly(img, cv_contours, (255, 255, 255))
    return img


def get_needle1(img):
    """获取采血针图像
    :param img: 输入图像
    :return: 输出图像
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 80), (255, 255, 255))  # (0,0,157)
    mask = cv2.bitwise_not(mask)
    needle1 = cv2.bitwise_and(img, img, mask=mask)
    background = np.zeros(img.shape, img.dtype)
    background[:, :, :] = 255
    mask = cv2.bitwise_not(mask)
    dst = cv2.bitwise_or(needle1, background, mask=mask)

    return dst


def get_needle2(src):
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    ret2, dst2 = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)
    return dst2


def needle_clear_background(src):
    """漫水填充去除背景（采血针）
    :param src:
    :return:
    """
    # img = src[2:src.shape[0] - 2, 0:src.shape[1]]  # [高 ，宽]
    # blur_img = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)  # 灰度图

    height, width = src.shape  # 获取图片宽高
    # (_, blur_img) = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)  # 二值化 固定阈值127

    # 去除黑色背景，seedPoint代表初始种子，进行四次，即对四个角都做一次，可去除最外围的黑边
    blur_img = cv2.floodFill(src, mask=None, seedPoint=(width - 3, 3), newVal=(255, 255, 255))[1]  # 从右侧漫水填充
    blur_img = cv2.floodFill(src, mask=None, seedPoint=(5, 5), newVal=(255, 255, 255))[1]  # 从左侧漫水填充
    blur_img = cv2.floodFill(src, mask=None, seedPoint=(3, height - 1), newVal=(255, 255, 255))[1]  #从左下角漫水填充
    blur_img = cv2.floodFill(src, mask=None, seedPoint=(width - 1, height - 1), newVal=(255, 255, 255))[1] #从右下角漫水填充
    return blur_img


def get_needle_line(img, src):
    """hough直线检测
    :param img:
    :param src:
    :return:
    """
    img = cv2.GaussianBlur(img, (3, 3), 0)
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 120)
    for line in lines:
        rho = line[0][0]
        theta = line[0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(src, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # 经验参数
    # minLineLength = 200
    # maxLineGap = 40
    # lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 29, minLineLength, maxLineGap)
    # for x1, y1, x2, y2 in lines[0]:
    #     cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return src


def thin_needle(img):
    """
    细化算法（采血针）
    :param img:
    :return:
    """
    img[img > 127] = 127
    img[img < 127] = 1
    img[img == 127] = 0
    thin_img = thin(img)
    thin_img = thin_img.astype(np.uint8) * 255
    return thin_img


def needle_denoise(img, n_openside, n_closedside, n_blurside):
    """
    开闭运算去除噪声（采血针）
    :param img: 输入图像
    :param n_openside: 开运算模板边长
    :param n_closedside: 闭运算模板边长
    :param n_blurside: 中值滤波模板边长
    :return:输出图像
    """
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    blur = cv2.medianBlur(img, n_blurside)
    # openkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (n_openside, n_openside))
    # # 开运算 先腐蚀后膨胀 消除暗背景下的较亮区域、平滑边界、纤细处分离物体
    # opening = cv2.morphologyEx(blur, cv2.MORPH_OPEN, openkernel)
    closedkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (n_closedside, n_closedside))
    # 闭运算 先膨胀后腐蚀 填充白色区域里细小黑色空洞的区域 连接临近物体
    closed = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, closedkernel)
    denoise_img = cv2.cvtColor(closed, cv2.COLOR_BGR2GRAY)
    return denoise_img


def get_needle_point(img):
    """
    获取采血针针尖坐标
    :param img: 输入图像
    :return: 输出坐标点
    """
    img_info = img.shape
    cv2.rectangle(img, (0, 0), (int(img_info[0]*0.3), img_info[1]), (0, 0, 0), 2)
    cv2.rectangle(img, (img_info[0]-10, 0), (img_info[0], img_info[1]), (0, 0, 0), 2)
    row, col = np.where(img == 255)

    return (row[-1], col[-1])


def get_line(r_binary, step_theta=2, step_rho=2):
    """
    直线检测（血管与采血针）
    :param r_binary: 输入二值化图像
    :param step_theta:
    :param step_rho:
    :return:
    """
    index = np.where(r_binary == 255)
    rows, cols = r_binary.shape
    l = round(math.sqrt(pow(rows - 1, 2.0) + pow(cols - 1, 2.0))) + 1
    num_theat = int(180.0/step_theta)
    num_rho = int(2 * l / step_rho + 1)
    accumulator = np.zeros((num_rho, num_theat), np.uint8)
    acc_dict = {}
    for k1 in range(num_rho):
        for k2 in range(num_theat):
            acc_dict[(k1, k2)] = []
    for i in range(len(index[0])):
        row = index[0][i]
        col = index[1][i]
        for m in range(num_theat):
            rho = col * math.cos(step_theta * m / 180.0 * math.pi) + row * math.sin(step_theta * m / 180.0 * math.pi)
            n = int(round(rho + l)/step_rho)
            accumulator[n, m] += 1
            acc_dict[(n, m)].append((row, col))
    location = np.where(accumulator == np.max(accumulator))
    point = acc_dict[(location[0][0], location[1][0])]
    if point:
        line = [point[0], point[len(point) - 1]]
    else:
        line = None
    return line


def get_theta(line):
    """
    获取直线的角度
    :param line:
    :return:
    """
    x1, y1 = line[0][0], line[0][1]
    x2, y2 = line[1][0], line[1][1]

    angle = math.atan2((y2 - y1), (x2 - x1))
    theta = angle * (180.0 / math.pi)
    return theta


def cmp_theta(blood_theta,needle_theta):
    """
    比较两直线的角度，判断是否一致
    :param blood_theta: 血管所在直线的角度
    :param needle_theta: 采血针所在直线的角度
    :return: 角度一致返回True，否则返回False
    """
    # if blood_theta == needle_theta :
    if abs(blood_theta - needle_theta)<3:
        return True
    return False

def pil2np(src):
    src = np.asarray(src)
    return src

def np2pil(src):
    src = fromarray(np.uint8(src))
    return src

def show_img(src):
    img = np2pil(src)
    plt.imshow(img,'gray')
    plt.show()
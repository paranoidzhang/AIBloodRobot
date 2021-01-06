import cv2


if __name__ == "__main__":
    src = cv2.imread("F:/2projectInfo/blood_robot/original/needle3.png")
    gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)

    blur = cv2.medianBlur(gray, 5)
    # temp, dst1 = cv2.threshold(blur,80,255,cv2.THRESH_BINARY)
    ret2, dst2 = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)


    cv2.imshow("biimg", dst2)
    cv2.imshow("src", src)
    cv2.waitKey(0)


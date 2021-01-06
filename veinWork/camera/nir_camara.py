import cv2
from veinWork.utils.config_util import Config

cv_conf = Config()


def init_nir_img():
    try:
        cap = cv2.VideoCapture(cv_conf.get_nir_camera_config()['device'])
        return cap
    except Exception as ex:
        print("nir_camera capture image failed")


def get_nir_img(cap):
    while True:
        try:
            ret, src = cap.read()
            return src
        except Exception as ex:
            print("nir_camera capture image failed")


if __name__ == "__main__":

    cap = init_nir_img()
    while 1:
        src = get_nir_img(cap)
        cv2.imshow("src",src)
        cv2.waitKey(10)
    cv2.destroyAllWindows()
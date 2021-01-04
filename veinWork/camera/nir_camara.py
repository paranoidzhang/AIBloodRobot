import cv2
from veinWork.utils.config_util import Config

cv_conf = Config()


def init_nir_img():
    while True:
        try:
            cap = cv2.VideoCapture(cv_conf.get_nir_camera_config()['device'])
            return cap
        except Exception as ex:
            print("nir_camera capture image failed")
            raise ex


def get_nir_img(cap):
    while True:
        try:
            ret, src = cap.read()
            return src
        except Exception as ex:
            print("nir_camera capture image failed")


if __name__ == "__main__":
    cap = init_nir_img()
    src = get_nir_img(cap)
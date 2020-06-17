import numpy as np
import cv2
import base64


def get_ndarray_by_bytes(img):
    img = base64.b64decode(img)
    img_np_array = np.frombuffer(img, np.uint8)
    return cv2.imdecode(img_np_array, cv2.IMREAD_COLOR)
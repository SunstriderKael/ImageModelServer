import os
from PIL import Image
import numpy as np
import copy
import cv2
import shutil
import random


def divide_train_test(img_folder, desc_folder, ratio=0.1):
    img_list = get_img_list(img_folder)
    for img in img_list:
        rand = random.random()
        img_path = os.path.join(img_folder, img)
        if rand < ratio:
            target_folder = os.path.join(desc_folder, 'test')
        else:
            target_folder = os.path.join(desc_folder, 'train')
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        desc_path = os.path.join(target_folder, img)
        shutil.copy(img_path, desc_path)


def copy_img_infolder(src_folder, desc_folder):
    img_list = get_img_list(src_folder)
    for img in img_list:
        src_img = os.path.join(src_folder, img)
        desc_img = os.path.join(desc_folder, img)
        shutil.copy(src_img, desc_img)


def get_img_list(img_folder):
    img_list = list()
    for root, dirs, files in os.walk(img_folder):
        for file in files:
            file_path = os.path.join(root, file)
            if is_valid_img(file_path):
                img_list.append(file)
    return img_list


def is_valid_img(img_path):
    try:
        Image.open(img_path).verify()
        img = Image.open(img_path)
        if img is None:
            return False
        height = img.size[0]
        width = img.size[1]
        if height == 0 or width == 0:
            return False
    except:
        return False
    return True


def cv2pil(img):
    tmp_img = copy.deepcopy(img)
    b = tmp_img[:, :, 0]
    g = tmp_img[:, :, 1]
    r = tmp_img[:, :, 2]
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b

    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    return img


def pil2cv(img):
    img = np.array(img)
    tmp_img = copy.deepcopy(img)
    r = tmp_img[:, :, 0]
    g = tmp_img[:, :, 1]
    b = tmp_img[:, :, 2]

    img[:, :, 0] = b
    img[:, :, 1] = g
    img[:, :, 2] = r
    return img


def padding(img, size):
    (height, width) = size
    img_height = img.shape[0]
    img_width = img.shape[1]
    channel = img.shape[3]

    img_whratio = img_width / img_height
    whratio = width / height
    output_img = np.zeros((height, width, channel))
    # pad top and bottom
    if img_whratio > whratio:
        resize_ratio = width / img_width
        resize_height = img_height * resize_ratio
        resize_img = cv2.resize(img, (width, resize_height))
        pad_top = int((resize_height - img_height) / 2)
        pad_left = 0
        output_img[pad_top: pad_top + resize_height, :, :] = resize_img

    else:
        resize_ratio = height / img_height
        resize_width = img_width * resize_ratio
        resize_img = cv2.resize(img, (resize_width, height))
        pad_top = 0
        pad_left = int((resize_width - img_width) / 2)
        output_img[:, pad_left:pad_left + resize_width, :] = resize_img
    return output_img, pad_top, pad_left


def __order_points(pts):
    # 初始化坐标点
    rect = np.zeros((4, 2), dtype="float32")

    # 获取左上角和右下角坐标点
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # 分别计算左上角和右下角的离散差值
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    """
    opencv 4点放射变换
    ————————————————
    版权声明：本文为CSDN博主「技术挖掘者」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
    原文链接：https://blog.csdn.net/wzz18191171661/article/details/99174861
    :param image:输入图片
    :param pts:原始目标的4个坐标点（左上，右上，右下，左下）
    :return:
    """
    # 获取坐标点，并将它们分离开来
    rect = __order_points(pts)
    (tl, tr, br, bl) = rect

    # 计算新图片的宽度值，选取水平差值的最大值
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # 计算新图片的高度值，选取垂直差值的最大值
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # 构建新图片的4个坐标点
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # 获取仿射变换矩阵并应用它
    M = cv2.getPerspectiveTransform(rect, dst)
    # 进行仿射变换
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # 返回变换后的结果
    return warped


if __name__ == '__main__':
    img_path = '/home/gaoyuanzi/Documents/test_model_helper/test_transform.jpg'
    save_img_path = '/home/gaoyuanzi/Documents/test_model_helper/four_point_transform.jpg'
    pts = np.array([(73, 239), (356, 117), (475, 265), (187, 443)])

    img = cv2.imread(img_path)
    warped = four_point_transform(img, pts)
    cv2.imwrite(save_img_path, warped)

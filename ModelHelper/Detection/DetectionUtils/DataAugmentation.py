from ModelHelper.Detection.DetectionUtils.Quadrangle import flip as qua_flip
from ModelHelper.Detection.DetectionUtils.Quadrangle import resize as qua_resize
from ModelHelper.Detection.DetectionUtils.Quadrangle import txt2array, array2txt
from ModelHelper.Common.CommonUtils.HandleImage import four_point_transform
from ModelHelper.Detection.DetectionUtils import draw_infolder
import random
import numpy as np
import cv2


class GaussianBlur:
    def __init__(self, kernel_size=(3, 3), sigma=1, aug_ratio=0.3):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.aug_ratio = aug_ratio

    def __call__(self, img, qua_array):
        aug_ratio = random.random()
        if aug_ratio < self.aug_ratio:
            img = cv2.GaussianBlur(img, self.kernel_size, self.sigma)
        return img, qua_array


class SaltNoise:
    def __init__(self, ratio=0.1, aug_ratio=0.3):
        self.ratio = ratio
        self.aug_ratio = aug_ratio

    def __call__(self, img, qua_array):
        aug_ratio = random.random()
        if aug_ratio < self.aug_ratio:
            h, w, d = img.shape
            for row in range(h):
                for col in range(w):
                    noise_ratio = random.random()
                    if noise_ratio < self.ratio:
                        color_flag = random.random()
                        if color_flag < 0.5:
                            img[row, col, :] = 0
                        else:
                            img[row, col, :] = 255
        return img, qua_array


class RandomDistort:
    def __init__(self, distort_num=10, ratio=0.05, aug_ratio=0.3):
        self.distort_num = distort_num
        if distort_num <= 1:
            raise RuntimeError('distort_num should > 1!')
        if ratio >= 0.5:
            raise RuntimeError('ratio should < 0.5!')
        self.ratio = ratio

        self.aug_ratio = aug_ratio

    def __call__(self, img, qua_array):
        aug_ratio = random.random()
        if aug_ratio < self.aug_ratio:
            height = img.shape[0]
            width = img.shape[1]
            distort_point_num = self.distort_num - 1

            distort_gap = width // self.distort_num
            gap_array = np.zeros((self.distort_num,))
            gap_array[:] = distort_gap
            yushu = width - distort_gap * self.distort_num
            gap_array[0:yushu] += 1

            rand_w_list = list()
            for idx in range(distort_point_num):
                rand_w = (2 * random.random() - 1) * self.ratio * distort_gap
                rand_w_list.append(rand_w)

            distort_top_pos_list = list()
            distort_top_pos_list.append((0, 0))
            for idx in range(distort_point_num):
                previous_x = distort_top_pos_list[idx][0]
                pos_x = previous_x + gap_array[idx] + rand_w_list[idx]
                rand_h = random.random() * self.ratio * height
                distort_top_pos_list.append((pos_x, rand_h))
            distort_top_pos_list.append((width - 1, 0))

            distort_bottom_pos_list = list()
            distort_bottom_pos_list.append((0, height - 1))
            for idx in range(distort_point_num):
                previous_x = distort_bottom_pos_list[idx][0]
                pos_x = previous_x + gap_array[idx] + rand_w_list[idx]
                rand_h = random.random() * self.ratio * height
                distort_bottom_pos_list.append((pos_x, height - 1 - rand_h))
            distort_bottom_pos_list.append((width - 1, height - 1))

            distort_img = np.zeros((height, 1, 3), dtype=np.uint8)
            for idx in range(self.distort_num):
                affine = np.zeros((4, 2))
                point1 = distort_top_pos_list[idx]
                point2 = distort_top_pos_list[idx + 1]
                point3 = distort_bottom_pos_list[idx]
                point4 = distort_bottom_pos_list[idx + 1]
                affine[0][0] = point1[0]
                affine[0][1] = point1[1]
                affine[1][0] = point2[0]
                affine[1][1] = point2[1]
                affine[2][0] = point3[0]
                affine[2][1] = point3[1]
                affine[3][0] = point4[0]
                affine[3][1] = point4[1]
                distort_patch = four_point_transform(img, affine)
                patch_width = int(gap_array[idx])
                distort_patch = cv2.resize(distort_patch, (patch_width, height))
                distort_img = np.concatenate([distort_img, distort_patch], axis=1)
            img = distort_img[:, 1:width + 1]
        return img, qua_array


class DetectionCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, qua_array):
        for t in self.transforms:
            img, qua_array = t(img, qua_array)
        return img, qua_array


class Resize:
    def __init__(self, size):
        (self.height, self.width) = size

    def __call__(self, img, qua_array):
        (h, w) = img.shape[:2]
        img = cv2.resize(img, dsize=(self.width, self.height))
        ratio_h = self.height / h
        ratio_w = self.width / w
        qua_array = qua_resize(qua_array, (ratio_h, ratio_w))
        return img, qua_array


class RandomCrop:
    def __init__(self, resize_ratio, size):
        self.resize_ratio = resize_ratio
        self.size = size

    def __call__(self, img, qua_array):
        img_height = img.shape[0]
        img_width = img.shape[1]
        if qua_array.shape[0] == 0:
            left = int(0.25 * img_width)
            right = int(0.75 * img_width)
            top = int(0.25 * img_height)
            bottom = int(0.75 * img_height)
        else:
            left = int(qua_array[:, :, 0].min())
            right = int(qua_array[:, :, 0].max())
            bottom = int(qua_array[:, :, 1].max())
            top = int(qua_array[:, :, 1].min())

        center_x = int((left + right) / 2)
        center_y = int((top + bottom) / 2)

        width = right - left
        height = bottom - top

        if width > height:
            crop_left = int(center_x - width / 2)
            crop_top = int(center_y - width / 2)
            crop_right = int(crop_left + width)
            crop_bottom = int(crop_top + width)

        else:
            crop_left = int(center_x - height / 2)
            crop_top = int(center_y - height / 2)
            crop_right = int(crop_left + height)
            crop_bottom = int(crop_top + height)

        crop_width = crop_right - crop_left
        crop_height = crop_bottom - crop_top
        (min_ratio, max_ratio) = self.resize_ratio
        assert max_ratio >= min_ratio >= 1
        expand_ratio = random.uniform(min_ratio, max_ratio)
        left_pad = int((expand_ratio - 1) * random.random() * crop_width)
        top_pad = int((expand_ratio - 1) * random.random() * crop_height)

        adj_left = left_pad - crop_left
        adj_top = top_pad - crop_top

        crop_left = crop_left - left_pad
        crop_top = crop_top - top_pad
        crop_right = crop_left + int(expand_ratio * crop_width)
        crop_bottom = crop_top + int(expand_ratio * crop_height)

        if crop_left < 0:
            img_left = 0
            crop_left = abs(crop_left)
        else:
            img_left = crop_left
            crop_left = 0

        if crop_top < 0:
            img_top = 0
            crop_top = abs(crop_top)
        else:
            img_top = crop_top
            crop_top = 0

        if crop_right > img_width:
            img_right = img_width
            crop_right = (img_right - img_left) + crop_left
        else:
            img_right = crop_right
            crop_right = (img_right - img_left) + crop_left

        if crop_bottom > img_height:
            img_bottom = img_height
            crop_bottom = (img_bottom - img_top) + crop_top
        else:
            img_bottom = crop_bottom
            crop_bottom = (img_bottom - img_top) + crop_top

        crop = np.zeros((int(expand_ratio * crop_height), int(expand_ratio * crop_width), 3))
        crop[crop_top:crop_bottom:, crop_left:crop_right, :] = img[img_top:img_bottom, img_left:img_right, :]

        width_ratio = self.size / crop.shape[1]
        height_ratio = self.size / crop.shape[0]
        crop = cv2.resize(crop, (self.size, self.size))

        if qua_array.shape[0] > 0:
            qua_array[:, :, 0] = (qua_array[:, :, 0] + adj_left) * width_ratio
            qua_array[:, :, 1] = (qua_array[:, :, 1] + adj_top) * height_ratio

            qua_array[qua_array < 0] = 0
            qua_array[qua_array[:, :, 0] >= self.size] = self.size - 1
            qua_array[qua_array[:, :, 1] >= self.size] = self.size - 1
        return crop, qua_array


class CenterRotate:
    def __init__(self, angle_threshold):
        self.angle_threshold = angle_threshold

    def __rotate_bound(self, img, angle):
        h = img.shape[0]
        w = img.shape[1]
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        return cv2.warpAffine(img, M, (nW, nH))

    def __rotate_box(self, bb, cx, cy, h, w, theta):
        new_bb = list(bb)

        for i, coord in enumerate(bb):
            M = cv2.getRotationMatrix2D((cx, cy), theta, 1.0)
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])

            nW = int(h * sin) + (w * cos)
            nH = int(h * cos) + (w * sin)

            M[0, 2] += (nW / 2) - cx
            M[1, 2] += (nH / 2) - cy

            v = [coord[0], coord[1], 1]
            calculated = np.dot(M, v)
            new_bb[i] = (calculated[0], calculated[1])
        return new_bb

    def __call__(self, img, qua_array):
        (min_angle, max_angle) = self.angle_threshold
        angle = random.uniform(min_angle, max_angle)

        (h, w) = img.shape[:2]
        cx = w / 2
        cy = h / 2
        img = self.__rotate_bound(img, angle)

        if qua_array.shape[0] > 0:
            new_qua = list()
            qua_num = qua_array.shape[0]
            for idx in range(qua_num):
                qua = qua_array[idx, :, :]
                qua = self.__rotate_box(qua, cx, cy, h, w, angle)
                new_qua.append(qua)
            qua_array = np.array(new_qua)
        return img, qua_array


class CenterRotateNoAngle:
    """
    中心旋转之后对取标注框的最小外切矩形
    """

    def __init__(self, angle_threshold):
        self.angle_threshold = angle_threshold

    def __rotate_bound(self, img, angle):
        h = img.shape[0]
        w = img.shape[1]
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        return cv2.warpAffine(img, M, (nW, nH))

    def __rotate_box(self, bb, cx, cy, h, w, theta):
        new_bb = list(bb)

        for i, coord in enumerate(bb):
            M = cv2.getRotationMatrix2D((cx, cy), theta, 1.0)
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])

            nW = int(h * sin) + (w * cos)
            nH = int(h * cos) + (w * sin)

            M[0, 2] += (nW / 2) - cx
            M[1, 2] += (nH / 2) - cy

            v = [coord[0], coord[1], 1]
            calculated = np.dot(M, v)
            new_bb[i] = (calculated[0], calculated[1])
        return new_bb

    def __call__(self, img, qua_array):
        (min_angle, max_angle) = self.angle_threshold
        angle = random.uniform(min_angle, max_angle)

        (h, w) = img.shape[:2]
        cx = w / 2
        cy = h / 2
        img = self.__rotate_bound(img, angle)

        if qua_array.shape[0] > 0:
            new_qua = list()
            qua_num = qua_array.shape[0]
            for idx in range(qua_num):
                qua = qua_array[idx, :, :]
                qua = self.__rotate_box(qua, cx, cy, h, w, angle)
                qua = np.array(qua)
                left = int(qua[:, 0].min())
                right = int(qua[:, 0].max())
                top = int(qua[:, 1].min())
                bottom = int(qua[:, 1].max())
                qua[0, 0] = left
                qua[0, 1] = top
                qua[1, 0] = right
                qua[1, 1] = top
                qua[2, 0] = right
                qua[2, 1] = bottom
                qua[3, 0] = left
                qua[3, 1] = bottom
                new_qua.append(qua)
            qua_array = np.array(new_qua)
        return img, qua_array


class Flip:
    def __init__(self, flip_type, ratio):
        assert flip_type in ['Horizontal', 'Vertical']
        if flip_type == 'Horizontal':
            self.type = 1
        elif flip_type == 'Vertical':
            self.type = 0
        self.ratio = ratio

    def __call__(self, img, qua_array):
        rand = random.random()
        if rand < self.ratio:
            img = cv2.flip(img, self.type)
            if qua_array.shape[0] > 0:
                qua_array = qua_flip(qua_array, self.type, img.shape)
            return img, qua_array
        else:
            return img, qua_array


class Padding:
    def __init__(self, size):
        self.size = size

    def __call__(self, img, qua_array):
        (output_h, output_w) = self.size
        img_h = img.shape[0]
        img_w = img.shape[1]
        img_c = img.shape[2]

        output = np.zeros((output_h, output_w, img_c))
        if img_w / img_h > output_w / output_h:
            ratio = output_w / img_w
            resize_h = int(ratio * img_h)
            pad_top = int((output_h - resize_h) / 2)
            pad_left = 0
            img = cv2.resize(img, (output_w, resize_h))
            output[pad_top: pad_top + resize_h, :, :] = img
        else:
            ratio = output_h / img_h
            resize_w = int(ratio * img_w)
            pad_top = 0
            pad_left = int((output_w - resize_w) / 2)
            img = cv2.resize(img, (resize_w, output_h))
            output[:, pad_left:pad_left + resize_w, :] = img
        if qua_array.shape[0] > 0:
            qua_array = qua_array.astype(np.float64)
            qua_array *= ratio
            qua_array = qua_array.astype(np.int32)
            qua_array[:, :, 0] += pad_left
            qua_array[:, :, 1] += pad_top

        return output, qua_array


if __name__ == '__main__':
    transforms = DetectionCompose([
        Padding(768)
    ])

    img_path = '/home/gaoyuanzi/Documents/test_model_helper/data/test_aug/input/1.jpg'
    txt_path = '/home/gaoyuanzi/Documents/test_model_helper/data/test_aug/input/1.txt'
    img = cv2.imread(img_path)
    array = txt2array(txt_path)

    img, array = transforms(img, array)
    desc_img = '/home/gaoyuanzi/Documents/test_model_helper/data/test_aug/output/1.jpg'
    desc_txt = '/home/gaoyuanzi/Documents/test_model_helper/data/test_aug/output/1.txt'
    cv2.imwrite(desc_img, img)
    array2txt(array, desc_txt)
    draw_infolder('/home/gaoyuanzi/Documents/test_model_helper/data/test_aug/output')

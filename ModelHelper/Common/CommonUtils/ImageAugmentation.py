from ModelHelper.Common.CommonUtils.HandleImage import four_point_transform
from ModelHelper.Common.CommonUtils.Wrapper import cv_fit_pil
import numpy as np
import cv2
import random
from math import fabs, sin, cos, radians


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class RGB2GRY:
    def __init__(self):
        pass

    @cv_fit_pil
    def __call__(self, img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


class AutoLevelsAdjustment:
    def __init__(self):
        pass

    # @cv_fit_pil
    def __call__(self, img):
        h, w, d = img.shape
        newimg = np.zeros([h, w, d])
        for i in range(d):
            imghist = self.__compute_hist(img[:, :, i])
            minlevel = self.__compute_min_level(imghist, 8.3, h * w)
            maxlevel = self.__compute_max_level(imghist, 2.2, h * w)
            newmap = self.__linear_map(minlevel, maxlevel)
            if newmap is None:
                return img
            for j in range(h):
                newimg[j, :, i] = newmap[img[j, :, i]]
        return newimg

    @staticmethod
    def __compute_hist(img):
        h, w = img.shape
        hist, bin_edge = np.histogram(img.reshape(1, w * h), bins=list(range(257)))
        return hist

    @staticmethod
    def __compute_min_level(hist, rate, pnum):
        sum = 0
        for i in range(256):
            sum += hist[i]
            if sum >= (pnum * rate * 0.01):
                return i

    @staticmethod
    def __compute_max_level(hist, rate, pnum):
        sum = 0
        for i in range(256):
            sum += hist[255 - i]
            if sum >= (pnum * rate * 0.01):
                return 255 - i

    @staticmethod
    def __linear_map(minlevel, maxlevel):
        if minlevel >= maxlevel:
            return None
        else:
            newmap = np.zeros(256)
            for i in range(256):
                if i < minlevel:
                    newmap[i] = 0
                elif i > maxlevel:
                    newmap[i] = 255
                else:
                    newmap[i] = (i - minlevel) / (maxlevel - minlevel) * 255
            return newmap


class GaussianBlur:
    def __init__(self, kernel_size=(3, 3), sigma=1, aug_ratio=0.3):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.aug_ratio = aug_ratio

    @cv_fit_pil
    def __call__(self, img):
        aug_ratio = random.random()
        if aug_ratio < self.aug_ratio:
            img = cv2.GaussianBlur(img, self.kernel_size, self.sigma)
        return img


class SaltNoise:
    def __init__(self, ratio=0.1, aug_ratio=0.3):
        self.ratio = ratio
        self.aug_ratio = aug_ratio

    @cv_fit_pil
    def __call__(self, img):
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
        return img


class RandomRotate:
    def __init__(self, angle=(-180, 180), aug_ratio=0.3):
        self.angle = angle
        self.aug_ratio = aug_ratio

    @cv_fit_pil
    def __call__(self, img):
        aug_ratio = random.random()
        if aug_ratio < self.aug_ratio:
            random_angle = random.randint(self.angle[0], self.angle[1])

            height, width = img.shape[:2]
            height_new = int(width * fabs(sin(radians(random_angle))) + height * fabs(cos(radians(random_angle))))
            width_new = int(height * fabs(sin(radians(random_angle))) + width * fabs(cos(radians(random_angle))))
            mat_rotation = cv2.getRotationMatrix2D((width / 2, height / 2), random_angle, 1)

            mat_rotation[0, 2] += (width_new - width) / 2
            mat_rotation[1, 2] += (height_new - height) / 2
            img = cv2.warpAffine(img, mat_rotation, (width_new, height_new))
        return img


class Padding:
    def __init__(self, size):
        if isinstance(size, tuple):
            self.size = size
        elif isinstance(size, int):
            self.size = (size, size)
        else:
            raise RuntimeError('size should be int or tuple!')

    @cv_fit_pil
    def __call__(self, img):
        img_h, img_w, img_c = img.shape
        (output_w, output_h) = self.size
        if img_w / img_h > output_w / output_h:
            ratio = output_w / img_w
            resize_w = output_w
            resize_h = int(img_h * ratio)
            top_pad = int((output_h - resize_h) / 2)
            left_pad = 0

        else:
            ratio = output_h / img_h
            resize_w = int(img_w * ratio)
            resize_h = output_h
            top_pad = 0
            left_pad = int((output_w - resize_w) / 2)
        output_img = np.zeros((output_h, output_w, img_c), dtype=np.uint8)
        img = cv2.resize(img, (resize_w, resize_h))
        output_img[top_pad:top_pad + resize_h, left_pad:left_pad + resize_w, :] = img
        return output_img


class RandomPadding:
    def __init__(self, top_pad, bottom_pad, left_pad, right_pad, aug_ratio=0.3):
        self.top_pad = top_pad
        self.bottom_pad = bottom_pad
        self.left_pad = left_pad
        self.right_pad = right_pad
        self.aug_ratio = aug_ratio

    @cv_fit_pil
    def __call__(self, img):
        aug_ratio = random.random()
        if aug_ratio < self.aug_ratio:
            top_pad = random.randint(0, self.top_pad)
            bottom_pad = random.randint(0, self.bottom_pad)
            left_pad = random.randint(0, self.left_pad)
            right_pad = random.randint(0, self.right_pad)

            height, width, channel = img.shape
            output = np.zeros((height + top_pad + bottom_pad, width + left_pad + right_pad, channel), dtype=np.uint8)
            output[top_pad:top_pad + height, left_pad:left_pad + width, :] = img
            img = cv2.resize(output, dsize=(width, height))
        return img


class RandomDistort:
    def __init__(self, distort_num=10, ratio=0.05, aug_ratio=0.3):
        self.distort_num = distort_num
        if distort_num <= 1:
            raise RuntimeError('distort_num should > 1!')
        if ratio >= 0.5:
            raise RuntimeError('ratio should < 0.5!')
        self.ratio = ratio

        self.aug_ratio = aug_ratio

    @cv_fit_pil
    def __call__(self, img):
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
        return img


class Denoise:
    def __init__(self, blur_kernel=(3, 3), blur_sigma=1, enhance_ratio=2):
        self.blur_kernel = blur_kernel
        self.blur_sigma = blur_sigma
        self.enhance_ratio = enhance_ratio

    @cv_fit_pil
    def __call__(self, img):
        img = img.astype(np.float64)
        blur_img = cv2.GaussianBlur(img, self.blur_kernel, self.blur_sigma)
        gap = img - blur_img
        gap_min = np.min(gap)
        gap_max = np.max(gap)
        gap_ratio = gap + abs(gap_min)
        gap_ratio = gap_ratio / (gap_max - gap_min)
        gap_ratio[gap_ratio[:, :, :] < 0.5] = 0
        denoise_img = gap * gap_ratio * self.enhance_ratio + blur_img
        denoise_img = np.clip(denoise_img, 0, 255)
        denoise_img = denoise_img.astype(np.uint8)
        return denoise_img


class LightCompensate:
    def __init__(self, block_size=10, blur_kernel=(3, 3), blur_sigma=1):
        """
        解决图像中光照不均衡的问题
        ————————————————
        版权声明：本文为CSDN博主「hudongloop」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
        原文链接：https://blog.csdn.net/u011276025/article/details/89790190
        :param block_size:
        :param blur_kernel:
        :param blur_sigma:
        """
        self.block_size = block_size
        self.blur_kernel = blur_kernel
        self.blur_sigma = blur_sigma

    def __call__(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        average = np.mean(gray)

        rows_new = int(np.ceil(gray.shape[0] / self.block_size))
        cols_new = int(np.ceil(gray.shape[1] / self.block_size))

        blockImage = np.zeros((rows_new, cols_new), dtype=np.float32)
        for r in range(rows_new):
            for c in range(cols_new):
                rowmin = r * self.block_size
                rowmax = (r + 1) * self.block_size
                if (rowmax > gray.shape[0]):
                    rowmax = gray.shape[0]
                colmin = c * self.block_size
                colmax = (c + 1) * self.block_size
                if (colmax > gray.shape[1]):
                    colmax = gray.shape[1]

                imageROI = gray[rowmin:rowmax, colmin:colmax]
                temaver = np.mean(imageROI)
                blockImage[r, c] = temaver

        blockImage = blockImage - average
        blockImage2 = cv2.resize(blockImage, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_CUBIC)
        gray2 = gray.astype(np.float32)
        dst = gray2 - blockImage2

        dst[dst > 255] = 255
        dst[dst < 0] = 0

        dst = dst.astype(np.uint8)
        dst = cv2.GaussianBlur(dst, self.blur_kernel, self.blur_sigma)
        dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

        return dst


if __name__ == '__main__':
    img_path = '/home/gaoyuanzi/Documents/test_model_helper/input.jpg'
    save_path = '/home/gaoyuanzi/Documents/test_model_helper/output.jpg'
    img = cv2.imread(img_path)
    lc = LightCompensate()
    img = lc(img)
    cv2.imwrite(save_path, img)

from ModelHelper.Common.CommonUtils import get
import numpy as np
import os
import cv2


class Quadrangle:
    def __init__(self, **kwargs):
        """
        pos_str: "x0,y0,x1,y1,x2,y2,x3,y3"
        pos_list: [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
        :param kwargs:
        """
        self.pos_str = get('pos_str', kwargs)
        self.pos_array = get('pos_array', kwargs)
        if self.pos_str is None and self.pos_array is None:
            raise RuntimeError('pos_str is None and pos_list is None!')
        if self.pos_str is None:
            self.pos_array = self.pos_array.astype(np.int32)
            self.pos_str = self.__pos_array2str()
        if self.pos_array is None:
            self.pos_array = self.__pos_str2pos_array()
            self.pos_array = self.pos_array.astype(np.int32)
        self.left = self.pos_array[:, 0].min()
        self.right = self.pos_array[:, 0].max()
        self.top = self.pos_array[:, 1].min()
        self.bottom = self.pos_array[:, 1].max()
        self.area = (self.bottom - self.top) * (self.right - self.left)

    def to_stander(self):
        """
        顺时针排列，第一个点为左上角
        :return: None
        """
        poly = np.array(self.pos_array)
        edge = [
            (poly[1][0] - poly[0][0]) * (poly[1][1] + poly[0][1]),
            (poly[2][0] - poly[1][0]) * (poly[2][1] + poly[1][1]),
            (poly[3][0] - poly[2][0]) * (poly[3][1] + poly[2][1]),
            (poly[0][0] - poly[3][0]) * (poly[0][1] + poly[3][1])
        ]
        poly_area = np.sum(edge) / 2.
        if abs(poly_area) < 1:
            print('invalid poly')
            self.pos_array = None
            self.pos_str = None
        else:
            # 保证poly 顺时针排列
            if poly_area > 0:
                # print('poly in wrong direction')
                poly = poly[(0, 3, 2, 1), :]
            score0 = np.sum(poly[0, :])
            score1 = np.sum(poly[1, :])
            score2 = np.sum(poly[2, :])
            score3 = np.sum(poly[3, :])
            # 保证poly 第一个坐标点为左上角
            min_score = min(score0, score1, score2, score3)
            if min_score == score0:
                pass
            elif min_score == score1:
                poly = poly[(1, 2, 3, 0), :]
            elif min_score == score2:
                poly = poly[(2, 3, 0, 1), :]
            elif min_score == score3:
                poly = poly[(3, 0, 1, 2,), :]
            self.pos_array = poly
            self.pos_str = self.__pos_array2str()

    def __pos_array2str(self):
        pos_str = str(self.pos_array[0][0]) + ',' + str(self.pos_array[0][1]) + ',' \
                  + str(self.pos_array[1][0]) + ',' + str(self.pos_array[1][1]) + ',' \
                  + str(self.pos_array[2][0]) + ',' + str(self.pos_array[2][1]) + ',' \
                  + str(self.pos_array[3][0]) + ',' + str(self.pos_array[3][1])
        return pos_str

    def __pos_str2pos_array(self):
        info = self.pos_str.split(',')
        if len(info) >= 8:
            x0 = int(float(info[0]))
            y0 = int(float(info[1]))
            x1 = int(float(info[2]))
            y1 = int(float(info[3]))
            x2 = int(float(info[4]))
            y2 = int(float(info[5]))
            x3 = int(float(info[6]))
            y3 = int(float(info[7]))

            pos_array = np.zeros((4, 2), dtype=np.int32)
            pos_array[0][0] = x0
            pos_array[0][1] = y0
            pos_array[1][0] = x1
            pos_array[1][1] = y1
            pos_array[2][0] = x2
            pos_array[2][1] = y2
            pos_array[3][0] = x3
            pos_array[3][1] = y3

            return pos_array
        else:
            print('quadrangle label should look like: "x0,y0,x1,y1,x2,y2,x3,y3", but the input is : {}'.format(
                self.pos_str))


def txt2list(txt_path, encoding='utf-8'):
    assert os.path.exists(txt_path)
    txt = open(txt_path, 'r', encoding=encoding)
    quadrangle_list = list()
    for line in txt.readlines():
        line = line.strip()
        line = line.replace('\n', '')
        if line == '':
            continue
        info = line.split(',')
        if len(info) >= 8:
            qua = Quadrangle(pos_str=line)
            quadrangle_list.append(qua)
        else:
            print('quadrangle label should look like: "x0,y0,x1,y1,x2,y2,x3,y3", but the input is : {}'.format(line))
    txt.close()
    return quadrangle_list


def list2txt(quadrangle_list, txt_path, encoding='utf-8'):
    if os.path.exists(txt_path):
        os.remove(txt_path)
    txt = open(txt_path, 'a', encoding=encoding)
    for qua in quadrangle_list:
        txt.write(qua.pos_str + '\n')
    txt.close()


def txt2array(txt_path, encoding='utf-8'):
    assert os.path.exists(txt_path)
    txt = open(txt_path, 'r', encoding=encoding)
    lines = txt.readlines()
    txt.close()
    filtered_lines = list()
    for line in lines:
        line = line.strip()
        line = line.replace('\n', '')
        if line == '':
            continue
        info = line.split(',')
        if len(info) >= 8:
            filtered_lines.append(line)
        else:
            print('quadrangle label should look like: "x0,y0,x1,y1,x2,y2,x3,y3", but the input is : {}'.format(line))
    qua_num = len(filtered_lines)
    quadrangle_array = np.zeros((qua_num, 4, 2), dtype=np.int32)
    idx = 0
    for line in filtered_lines:
        qua = Quadrangle(pos_str=line)
        quadrangle_array[idx, :, :] = qua.pos_array
        idx += 1

    return quadrangle_array


def array2txt(quadrangle_array, txt_path, encoding='utf-8'):
    if os.path.exists(txt_path):
        os.remove(txt_path)
    txt = open(txt_path, 'a', encoding=encoding)
    qua_num = quadrangle_array.shape[0]
    for idx in range(qua_num):
        qua = quadrangle_array[idx, :, :]
        qua = Quadrangle(pos_array=qua)
        txt.write(qua.pos_str + '\n')
    txt.close()


def compute_iou(qua1, qua2):
    top = min(qua1.top, qua2.top)
    left = min(qua1.left, qua2.left)
    bottom = max(qua1.bottom, qua2.bottom)
    right = max(qua1.right, qua2.right)

    qua1_array = qua1.pos_array
    qua1_array[:, 0] -= left
    qua1_array[:, 1] -= top
    qua2_array = qua2.pos_array
    qua2_array[:, 0] -= left
    qua2_array[:, 1] -= top

    qua1_draw = np.zeros((1, qua1_array.shape[0], 2), dtype=np.int32)
    qua1_draw[0, :, :] = qua1_array
    qua1_poly = np.zeros((bottom - top, right - left), dtype=np.int32)
    cv2.fillPoly(qua1_poly, qua1_draw, 1)
    # cv2.imwrite('qua1.png', qua1_poly*255)
    qua1_sum = qua1_poly.sum()

    qua2_draw = np.zeros((1, qua2_array.shape[0], 2), dtype=np.int32)
    qua2_draw[0, :, :] = qua2_array
    qua2_poly = np.zeros((bottom - top, right - left), dtype=np.int32)
    cv2.fillPoly(qua2_poly, qua2_draw, 1)
    # cv2.imwrite('qua2.png', qua2_poly*255)
    qua2_sum = qua2_poly.sum()

    inter = qua1_poly * qua2_poly
    # cv2.imwrite('inter.png', inter*255)

    inter = inter.sum()
    union = qua1_sum + qua2_sum - inter
    iou = inter / union
    return iou


def resize(qua_array, ratio):
    (ratio_h, ratio_w) = ratio
    qua_array[:, :, 0] *= ratio_w
    qua_array[:, :, 1] *= ratio_h
    return qua_array


def flip(qua_array, type, img_shape):
    """
    flip
    :param txt_path:
    :param type: type=0, vertical flip, type=1, horizontal flip
    :return:
    """

    assert type == 0 or type == 1
    (h, w) = img_shape[:2]
    if type == 0:
        qua_array[:, :, 1] = h - qua_array[:, :, 1]
    else:
        qua_array[:, :, 0] = w - qua_array[:, :, 0]

    return qua_array

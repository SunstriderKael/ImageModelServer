from ModelHelper.Common.CommonUtils.HandleImage import get_img_list, four_point_transform
from ModelHelper.Detection.DetectionUtils.Quadrangle import Quadrangle
from ModelHelper.Common.CommonUtils.HandleText import list2txt
import os
import cv2


def generate_sar_data(img_folder, desc_folder, label_path):
    if not os.path.exists(desc_folder):
        os.makedirs(desc_folder)
    img_list = get_img_list(img_folder)
    sar_label_list = list()
    for img in img_list:
        img_path = os.path.join(img_folder, img)
        txt_name = img.split('.')[0] + '.txt'
        txt_path = os.path.join(img_folder, txt_name)
        txt_file = open(txt_path, 'r', encoding='utf-8')
        idx = 0
        image = cv2.imread(img_path)
        for line in txt_file.readlines():
            info = line.split(',')
            x0 = info[0]
            y0 = info[1]
            x1 = info[2]
            y1 = info[3]
            x2 = info[4]
            y2 = info[5]
            x3 = info[6]
            y3 = info[7]
            label = info[8]

            pos_str = x0 + ',' + y0 + ',' + x1 + ',' + y1 + ',' + x2 + ',' + y2 + ',' + x3 + ',' + y3
            quadrangle = Quadrangle(pos_str=pos_str)
            quadrangle.to_stander()
            transformed_img = four_point_transform(image, quadrangle.pos_array)
            new_img = img.split('.')[0] + '_qua{}.jpg'.format(idx)
            idx += 1
            new_img_path = os.path.join(desc_folder, new_img)
            cv2.imwrite(new_img_path, transformed_img)
            print('generate sar data:{}'.format(new_img_path))

            sar_label = new_img_path+','+label
            sar_label_list.append(sar_label)
    list2txt(sar_label_list, label_path)

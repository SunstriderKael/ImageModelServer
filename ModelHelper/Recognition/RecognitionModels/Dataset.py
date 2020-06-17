from ModelHelper.Common.CommonUtils import get, get_valid
from ModelHelper.Common.CommonUtils.HandleImage import get_img_list
import os
import math
from random import shuffle
import random
import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class MultiDecoderClassifyDataset(Dataset):
    def __init__(self, **kwargs):
        self.folder = get_valid('folder', kwargs)
        self.is_training = get_valid('is_training', kwargs)
        self.class_dict = get_valid('class_dict', kwargs)
        self.transforms = get_valid('transforms', kwargs)
        self.size = get('size', kwargs, (32, 256, 3))
        self.split = get('split', kwargs, ',')
        self.index_list = self.__generate_index()

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        flag = True
        while flag:
            info = self.index_list[idx]
            info = info.strip('\n')
            img_info = info.split(self.split)
            img_path = img_info[0]
            label = img_info[1]
            # try:
            img = cv2.imread(img_path)
            img, valid_width = self.__padding(img)
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            img = self.transforms(img)
            img = img.float()
            label = int(label)
            flag = False
            # except Exception as e:
            #     print('Error occur on: {}'.format(img_path))
            #     print(e)
            #     idx = random.randint(0, len(self.index_list) - 1)
        return img, label

    def __padding(self, img):
        (height, width, channel) = self.size
        img_height = img.shape[0]
        img_width = img.shape[1]
        wh = width / height
        img_wh = img_width / img_height
        output = np.zeros(self.size, dtype=np.uint8)
        if img_wh > wh:
            ratio = width / img_width
            img_height = int(img_height * ratio)
            img = cv2.resize(img, (width, img_height))
            pad_top = int((height - img_height) / 2)
            output[pad_top:pad_top + img_height, :, :] = img
            valid_width = width
        else:
            ratio = height / img_height
            img_width = int(img_width * ratio)
            img = cv2.resize(img, (img_width, height))
            output[:, 0:img_width, :] = img
            valid_width = img_width
        return output, valid_width

    def __generate_index(self):
        index_list = list()
        max_len = 0
        for cls in self.class_dict:
            cls_path = os.path.join(self.folder, cls)
            cls_label = self.class_dict[cls]
            cls_index = self.__generate_cls_index(cls_path, cls_label)
            print('{} data num: {}'.format(cls, len(cls_index)))
            index_list.append(cls_index)
            if len(cls_index) > max_len:
                max_len = len(cls_index)

        if self.is_training:
            balance_index_list = self.__blance_index(index_list, max_len)
            return balance_index_list
        else:
            output_list = list()
            for index in index_list:
                output_list.extend(index)
            return output_list

    def __generate_cls_index(self, cls_path, cls_label):
        cls_label = str(cls_label)
        img_list = get_img_list(cls_path)
        index_list = list()
        for img in img_list:
            img_path = os.path.join(cls_path, img)
            index = img_path + self.split + cls_label
            index_list.append(index)
        return index_list

    def __blance_index(self, total_index_list, max_len):
        balance_index_list = list()
        for index_list in total_index_list:
            copy_ratio = math.ceil(float(max_len) / float(len(index_list)))
            index_num = 0
            for idx in range(copy_ratio):
                shuffle(index_list)
                for index in index_list:
                    if index_num < max_len:
                        index_num += 1
                        balance_index_list.append(index)
                    else:
                        break
        shuffle(balance_index_list)
        return balance_index_list


class SarDataset(Dataset):
    def __init__(self, **kwargs):
        self.index_path = get_valid('index_path', kwargs)
        self.folder = get_valid('folder', kwargs)
        self.transforms = get('transforms', kwargs, None)
        self.encoding = get('encoding', kwargs, 'UTF-8-sig')
        self.size = get('size', kwargs, (32, 256, 3))
        self.label_split = get('label_split', kwargs, ',')
        self.data_list = self.get_data_list()
        self.max_len = get('max_len', kwargs, 64)
        self.mask_ratio = get('mask_ratio', kwargs, (8, 4))

    def __len__(self):
        return len(self.data_list)

    def get_data_list(self):
        data_list = list()
        index = open(self.index_path, 'r', encoding=self.encoding)
        for line in index.readlines():
            try:
                line = line.replace('\n', '')
                line = line.strip()
                info = line.split(self.label_split)
                data = dict()
                img_name = info[0]
                img_path = os.path.join(self.folder, img_name)
                data['img_path'] = img_path
                data['label'] = info[1]
                data_list.append(data)
            except:
                print('error occur on: {}'.format(line))
                continue
        return data_list

    def __padding(self, img):
        (height, width, channel) = self.size
        img_height = img.shape[0]
        img_width = img.shape[1]
        wh = width / height
        img_wh = img_width / img_height
        output = np.zeros(self.size, dtype=np.uint8)
        if img_wh > wh:
            ratio = width / img_width
            img_height = int(img_height * ratio)
            img = cv2.resize(img, (width, img_height))
            pad_top = int((height - img_height) / 2)
            output[pad_top:pad_top + img_height, :, :] = img
            valid_width = width
        else:
            ratio = height / img_height
            img_width = int(img_width * ratio)
            img = cv2.resize(img, (img_width, height))
            output[:, 0:img_width, :] = img
            valid_width = img_width
        return output, valid_width

    def __getitem__(self, idx):
        data = self.data_list[idx]
        flag = True
        # while flag:
        #     try:
        img_path = data['img_path']
        label = data['label']
        img = cv2.imread(img_path)
        img, valid_width = self.__padding(img)
        if self.transforms is not None:
            img = self.transforms(img)
        # mask_h = int(self.size[0] / 8)
        # mask_w = int(self.size[1] / 4)

        mask_h = int(self.size[0] / self.mask_ratio[0])
        mask_w = int(self.size[1] / self.mask_ratio[1])

        mask = np.zeros((1, mask_h, mask_w))
        mask[:, :, :valid_width] = 1
        flag = False
        # except:
        #     print('error occur on: {}'.format(img_path))
        #     data_num = len(self.data_list)
        #     idx = random.randint(0, data_num)
        #     data = self.data_list[idx]

        return img, label, mask

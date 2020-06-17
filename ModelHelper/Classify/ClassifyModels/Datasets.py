from torch.utils.data import Dataset
from ModelHelper.Common.CommonUtils import get, get_valid
from ModelHelper.Common.CommonUtils.HandleImage import get_img_list
from PIL import Image
import os
import math
from random import shuffle
import random


class ClassifyDataset(Dataset):
    def __init__(self, **kwargs):
        self.folder = get_valid('folder', kwargs)
        self.is_training = get_valid('is_training', kwargs)
        self.class_dict = get_valid('class_dict', kwargs)
        self.transforms = get_valid('transforms', kwargs)
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
            try:
                img = Image.open(img_path)
                img = self.transforms(img)
                img = img.float()
                label = int(label)
                flag = False
            except Exception as e:
                print('Error occur on: {}'.format(img_path))
                print(e)
                idx = random.randint(0, len(self.index_list)-1)
        return img, label

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

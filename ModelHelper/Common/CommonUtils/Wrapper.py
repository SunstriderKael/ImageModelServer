from ModelHelper.Common.CommonUtils import get, get_valid
from ModelHelper.Common.CommonUtils.HandleText import list2txt
import time
import os
import datetime
import cv2
import PIL
from PIL import Image
import numpy


def time_consume(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        output = func(*args, **kwargs)
        end = time.time()
        print('{} use {} seconds!'.format(func.__name__, end - start))
        return output

    return wrapper


def config(func):
    def wrapper(*args, **kwargs):
        config_path = get('config_path', kwargs)
        if config_path is None:
            output_folder = get_valid('output_folder', kwargs)
            now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            output_folder = os.path.join(output_folder, now)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            kwargs['output_folder'] = output_folder
            config_path = os.path.join(output_folder, 'config.txt')
        avoid_key = get('avoid_key', kwargs, list())
        config_list = list()
        for key in kwargs:
            if key not in avoid_key:
                config = '{}:{}'.format(key, kwargs[key])
                print(config)
                config_list.append(config)
        list2txt(config_list, config_path)
        return func(*args, **kwargs)

    return wrapper


def cv_fit_pil(func):
    def wrapper(self, img):
        img = cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)
        img = func(self, img)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return img
    return wrapper

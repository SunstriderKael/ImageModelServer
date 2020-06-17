from ModelHelper.Common.CommonUtils.HandleImage import get_img_list
import requests
import base64
import json
import os
import cv2

TEST_URL = 'http://10.3.201.204:35600/retrieval/'


def test_add_feature(img_folder):
    url = TEST_URL + 'add_feature'
    img_name_list = get_img_list(img_folder)
    for img_name in img_name_list:
        img_path = os.path.join(img_folder, img_name)
        data = dict()
        data['img_name'] = img_name

        image = cv2.imread(img_path)
        image_str = cv2.imencode('.jpg', image)[1].tostring()
        b64_code = base64.b64encode(image_str)
        data['base64'] = b64_code.decode()
        data['info'] = img_name
        print(json.dumps(data))
        response = requests.post(url, data=json.dumps(data))
        print(response.text)


def test_retrieval(img_folder):
    url = TEST_URL + 'retrieval'
    img_list = get_img_list(img_folder)
    lib_id_list = [216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229]
    for img in img_list:
        img_path = os.path.join(img_folder, img)
        data = dict()
        data['img_name'] = img
        image = cv2.imread(img_path)
        image_str = cv2.imencode('.jpg', image)[1].tostring()
        b64_code = base64.b64encode(image_str)
        data['base64'] = b64_code.decode()
        data['top_n'] = 3
        data['lib_id_list'] = lib_id_list
        print(json.dumps(data))
        response = requests.post(url, data=json.dumps(data))
        print(response.text)


def test_global_retrieval(img_folder):
    url = TEST_URL + 'global_retrieval'
    img_list = get_img_list(img_folder)
    for img in img_list:
        img_path = os.path.join(img_folder, img)
        data = dict()
        data['img_name'] = img
        image = cv2.imread(img_path)
        image_str = cv2.imencode('.jpg', image)[1].tostring()
        b64_code = base64.b64encode(image_str)
        data['base64'] = b64_code.decode()
        data['top_n'] = 3
        print(json.dumps(data))
        response = requests.post(url, data=json.dumps(data))
        print(response.text)


if __name__ == '__main__':
    # test_add_feature(r'D:\gyz\ImageRetrieval\data\2020-05-12\fuben')
    # test_global_retrieval(r'D:\gyz\ImageRetrieval\data\2020-05-12\zhuben')
    test_retrieval(r'D:\gyz\ImageRetrieval\data\2020-05-12\zhuben')

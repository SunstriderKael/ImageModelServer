from ModelHelper.ImageRetrieval.Models import LSHRetrieval
from retrieval.models import RetrievalAntiFraud
from common.utils import get_ndarray_by_bytes
from retrieval.utils import feature2str, str2feature
from django.http import HttpResponse

from PIL import Image
import cv2
import json
import torch

# 初始化反欺诈图像检索模型
checkpoint = './weights/anti_fraud_retrieval/gl18-tl-resnet50-gem-w-83fdc30.pth'
anti_fraud_retrieval_model = LSHRetrieval(checkpoint=checkpoint, hash_size=0, input_dim=2048)


def add_feature(request):
    try:
        parameters = request.body.decode()
        parameters = json.loads(parameters)
        img_name = parameters['img_name']
        base64 = parameters['base64']
        img_info = parameters['info']

        # 参数异常处理
        try:
            image = get_ndarray_by_bytes(base64)
        except:
            output = {
                'code': 1100,
                'message': "Error, can't extract ndarray from {}'s base64;".format(img_name),
            }
            return HttpResponse(json.dumps(output))

        if img_info is None:
            img_info = ''

        if not isinstance(img_info, str):
            output = {
                'code': 1101,
                'message': "Error, the type of {}'s info is {} not str;".format(img_name, type(img_info)),
            }
            return HttpResponse(json.dumps(output))

        # 特征抽取
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if max(image.size) / min(image.size) > 5:
            output = {
                'code': 1102,
                'message': "Error, invalid image, max(image.size) / min(image.size) > 5;".format(img_name),
            }
            return HttpResponse(json.dumps(output))

        feature = anti_fraud_retrieval_model.extract_feature(image)
        feature = feature2str(feature)

        anti_fraud = RetrievalAntiFraud.objects.create(
            info=img_info,
            feature=feature,
            status=0
        )
        data = {
            'id': anti_fraud.id,
            'img_name': img_name,
        }

        output = {
            'message': 'Success',
            'data': data,
            'code': 200
        }

        return HttpResponse(json.dumps(output))
    except:
        output = {
            'message': 'Unknown error',
            'code': 1000
        }
        return HttpResponse(json.dumps(output))


def delete_feature():
    pass


def retrieval(request):
    try:
        parameters = request.body.decode()
        parameters = json.loads(parameters)
        lib_id_list = parameters['lib_id_list']
        img_name = parameters['img_name']
        img_bs64 = parameters['base64']
        top_n = parameters['top_n']

        # 构建lsh

        id_list = list()
        feature_list = list()

        for lib_id in lib_id_list:
            anti_fraud = RetrievalAntiFraud.objects.filter(id=lib_id).first()
            if anti_fraud is not None:
                feature = anti_fraud.feature
                feature = str2feature(feature)
                feature_list.append(feature)
                id = anti_fraud.id
                id_list.append(id)
            else:
                print("Warning, don't exist RetrievalAntiFraud.id = {}".format(lib_id))

        idx = 0
        feature_lib = torch.zeros(anti_fraud_retrieval_model.model.meta['outputdim'], len(feature_list))
        for feature in feature_list:
            feature_lib[:, idx] = feature
            idx += 1

        feature_dict = dict(zip(id_list, list(feature_lib.detach().cpu().numpy().T)))
        lsh = anti_fraud_retrieval_model.get_initial_lsh()
        for info, vec in feature_dict.items():
            lsh.index(vec.flatten(), extra_data=info)

        # 检索
        try:
            image = get_ndarray_by_bytes(img_bs64)
        except:
            output = {
                'code': 1100,
                'message': "Error, can't extract ndarray from {}'s base64;".format(img_name),
            }
            return HttpResponse(json.dumps(output))

        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if max(image.size) / min(image.size) > 5:
            output = {
                'code': 1102,
                'message': "Error, invalid image, max(image.size) / min(image.size) > 5;".format(img_name),
            }
            return HttpResponse(json.dumps(output))

        feature = anti_fraud_retrieval_model.extract_feature(image)
        response = lsh.query(feature.flatten(), num_results=top_n, distance_func="cosine")
        match_list = list()
        for idx in range(top_n):
            id = response[idx][0][1]
            score = response[idx][1]
            match = dict()
            match['id'] = id
            match['score'] = score
            match_list.append(match)

        data = {
            'img_name': img_name,
            'match_list': match_list,
        }

        output = {
            'message': 'Success',
            'data': data,
            'code': 200
        }
        return HttpResponse(json.dumps(output))
    except:
        output = {
            'message': 'Unknown error',
            'code': 1000
        }
        return HttpResponse(json.dumps(output))


def global_retrieval(request):
    try:
        parameters = request.body.decode()
        parameters = json.loads(parameters)
        img_name = parameters['img_name']
        img_bs64 = parameters['base64']
        top_n = parameters['top_n']

        # 构建lsh
        anti_fraud_list = RetrievalAntiFraud.objects.filter()
        id_list = list()
        feature_lib = torch.zeros(anti_fraud_retrieval_model.model.meta['outputdim'], len(anti_fraud_list))

        idx = 0
        for anti_fraud in anti_fraud_list:
            feature = anti_fraud.feature
            feature = str2feature(feature)
            feature_lib[:, idx] = feature
            id_list.append(anti_fraud.id)
            idx += 1

        feature_dict = dict(zip(id_list, list(feature_lib.detach().cpu().numpy().T)))
        lsh = anti_fraud_retrieval_model.get_initial_lsh()
        for info, vec in feature_dict.items():
            lsh.index(vec.flatten(), extra_data=info)

        # 检索
        try:
            image = get_ndarray_by_bytes(img_bs64)
        except:
            output = {
                'code': 1100,
                'message': "Error, can't extract ndarray from {}'s base64;".format(img_name),
            }
            return HttpResponse(json.dumps(output))

        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if max(image.size) / min(image.size) > 5:
            output = {
                'code': 1102,
                'message': "Error, invalid image, max(image.size) / min(image.size) > 5;".format(img_name),
            }
            return HttpResponse(json.dumps(output))

        feature = anti_fraud_retrieval_model.extract_feature(image)
        response = lsh.query(feature.flatten(), num_results=top_n, distance_func="cosine")
        match_list = list()
        for idx in range(top_n):
            id = response[idx][0][1]
            score = response[idx][1]
            match = dict()
            match['id'] = id
            match['score'] = score
            match_list.append(match)

        data = {
            'img_name': img_name,
            'match_list': match_list,
        }

        output = {
            'message': 'Success',
            'data': data,
            'code': 200
        }
        return HttpResponse(json.dumps(output))
    except:
        output = {
            'message': 'Unknown error',
            'code': 1000
        }
        return HttpResponse(json.dumps(output))

from classify.utils import init_card_9cls
from django.http import HttpResponse
from PIL import Image
from common.utils import get_ndarray_by_bytes
import cv2
import json
import time

# 初始化8类卡证分类模型
card_9cls_checkpoint = './weights/card_9cls/resnet18_epoch28_acc0.9989594172736732_loss0.0047.pth'
card_9cls_model, card_9cls_cls2code = init_card_9cls(card_9cls_checkpoint)


def card_9cls(request):
    try:
        parameters = request.body.decode()
        parameters = json.loads(parameters)
        img = parameters['img']
        img = get_ndarray_by_bytes(img)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        start = time.time()
        cls, score = card_9cls_model.predict(img=img)
        score = round(score, 4)
        end = time.time()

        cls = card_9cls_cls2code[cls]
        if score < 0.98:
            cls = 8

        pred = {
            'class': cls,
            'score': score,
            'cost_time': end - start,
        }

        data = {
            'code': 200,
            'message': 'success',
            'pred': pred
        }
    except:
        data = {
            'code': 1000,
            'message': 'unknown error',
            'pred': None
        }

    return HttpResponse(json.dumps(data))

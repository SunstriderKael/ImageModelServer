from ModelHelper.Classify.ClassifyModels.Predict import PredictNTSClassification


def init_card_9cls(checkpoint):
    class_map = {
        0: 'bank_card',
        1: 'driver_license_attachment',
        2: 'driver_license_master',
        3: 'id_card_back',
        4: 'id_card_front',
        5: 'others',
        6: 'vehicle_license_attachment_back',
        7: 'vehicle_license_attachment_front',
        8: 'vehicle_license_master'
    }
    model_name = 'Resnet18NTSClassifyModel'
    class_num = 9
    card_9cls_model = PredictNTSClassification(checkpoint=checkpoint, class_map=class_map, use_gpu=False,
                                               model_name=model_name, class_num=class_num)
    card_9cls_cls2code = {
        'bank_card': 0,
        'driver_license_master': 1,
        'driver_license_attachment': 2,
        'id_card_front': 3,
        'id_card_back': 4,
        'vehicle_license_master': 5,
        'vehicle_license_attachment_front': 6,
        'vehicle_license_attachment_back': 7,
        'others': 8
    }
    return card_9cls_model, card_9cls_cls2code

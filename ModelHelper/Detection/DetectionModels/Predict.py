from ModelHelper.Common.CommonModels.Predict import AbstractPredict
from ModelHelper.Common.CommonUtils import get, get_valid
from ModelHelper.Detection.DetectionModels.ModelFactory import DetectionModelFactory
from ModelHelper.Detection.DetectionModels.east_eval.eval_utils import detect, sort_poly

import torch
from torchvision import transforms
import numpy as np


class PredictEASTDetection(AbstractPredict):
    def __init__(self, **kwargs):
        super(PredictEASTDetection, self).__init__(**kwargs)
        model = self.init_model(**kwargs)
        model = self.load_model(model=model, **kwargs)
        if self.use_gpu:
            model = model.cuda()
        # model = DataParallel(model)
        self.model = model.eval()

    def init_model(self, **kwargs):
        return super(PredictEASTDetection, self).init_model(model_factory=DetectionModelFactory(), **kwargs)

    def load_model(self, **kwargs):
        return super(PredictEASTDetection, self).load_model(**kwargs)

    def pretreatment(self, **kwargs):
        eval_transforms = get('eval_transforms', kwargs)
        if eval_transforms is None:
            eval_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        img = get_valid('img', kwargs)
        img = img.astype(np.float32)
        return eval_transforms(img).unsqueeze(0)

    def predict(self, **kwargs):
        super(PredictEASTDetection, self).predict(**kwargs)
        score_map_thresh = get('score_map_thresh', kwargs, 0.7)
        box_thresh = get('box_thresh', kwargs, 0.1)
        nms_thresh = get('nms_thresh', kwargs, 0.2)

        img = self.pretreatment(**kwargs)
        with torch.no_grad():
            if self.use_gpu:
                img = img.cuda()
            score, geometry = self.model(image=img)

        score = score.permute(0, 2, 3, 1)
        geometry = geometry.permute(0, 2, 3, 1)
        score = score.data.cpu().numpy()
        geometry = geometry.data.cpu().numpy()
        boxes = detect(score_map=score,
                       geo_map=geometry,
                       score_map_thresh=score_map_thresh,
                       box_thresh=box_thresh,
                       nms_thresh=nms_thresh)

        pred_list = list()
        if boxes is not None:
            boxes = boxes[:, :8].reshape(-1, 4, 2)
            for box in boxes:
                box = sort_poly(box.astype(np.int32))
                pred_list.append(box)

        return pred_list

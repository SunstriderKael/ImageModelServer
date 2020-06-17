from ModelHelper.Common.CommonUtils import get, get_valid
from ModelHelper.Common.CommonModels.Predict import AbstractPredict
from ModelHelper.Common.CommonUtils.ImageAugmentation import Padding
from ModelHelper.Classify.ClassifyModels.ModelFactory import ClassifyModelFactory

import torch
from torch.nn import DataParallel
from torchvision import transforms


class PredictClassification(AbstractPredict):
    def __init__(self, **kwargs):
        super(PredictClassification, self).__init__(**kwargs)
        self.class_map = get_valid('class_map', kwargs)
        model = self.init_model(**kwargs)
        model = self.load_model(model=model, **kwargs)
        if self.use_gpu:
            model = model.cuda()
        model = DataParallel(model)
        self.model = model.eval()

    def init_model(self, **kwargs):
        return super(PredictClassification, self).init_model(model_factory=ClassifyModelFactory(), **kwargs)

    def load_model(self, **kwargs):
        return super(PredictClassification, self).load_model(**kwargs)

    def pretreatment(self, **kwargs):
        eval_transforms = get('eval_transforms', kwargs)
        if eval_transforms is None:
            input_size = get('input_size', kwargs, 600)
            crop_size = get('crop_size', kwargs, 448)

            eval_transforms = transforms.Compose([
                Padding(input_size),
                transforms.CenterCrop((crop_size, crop_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        img = get_valid('img', kwargs)
        return eval_transforms(img).float().unsqueeze(0)

    def predict(self, **kwargs):
        img = self.pretreatment(**kwargs)
        with torch.no_grad():
            if self.use_gpu:
                img = img.cuda()
            pred = self.model(image=img)
            _, pred_cls = torch.max(pred, 1)
            pred_cls = int(pred_cls)
            concat_logits = torch.nn.functional.softmax(pred, 1)
            score = float(concat_logits[0][pred_cls])
        return self.class_map[pred_cls], score


class PredictNTSClassification(PredictClassification):
    def __init__(self, **kwargs):
        super(PredictNTSClassification, self).__init__(**kwargs)

    def init_model(self, **kwargs):
        return super(PredictNTSClassification, self).init_model(**kwargs)

    def load_model(self, **kwargs):
        return super(PredictNTSClassification, self).load_model(**kwargs)

    def pretreatment(self, **kwargs):
        return super(PredictNTSClassification, self).pretreatment(**kwargs)

    def predict(self, **kwargs):
        img = self.pretreatment(**kwargs)
        with torch.no_grad():
            if self.use_gpu:
                img = img.cuda()
            _, concat_logits, _, _, _ = self.model(image=img)
            _, pred_cls = torch.max(concat_logits, 1)
            pred_cls = int(pred_cls)
            concat_logits = torch.nn.functional.softmax(concat_logits, 1)
            score = float(concat_logits[0][pred_cls])
        return self.class_map[pred_cls], score

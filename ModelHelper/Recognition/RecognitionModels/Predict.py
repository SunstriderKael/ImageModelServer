from ModelHelper.Common.CommonModels.Predict import AbstractPredict
from ModelHelper.Common.CommonUtils import get, get_valid
from ModelHelper.Recognition.RecognitionModels.ModelFactory import RecognitionModelFactory
from ModelHelper.Common.CommonUtils.ImageAugmentation import LightCompensate
from torchvision import transforms
import numpy as np
import torch
import cv2


class PredictSarRecognition(AbstractPredict):
    def __init__(self, **kwargs):
        super(PredictSarRecognition, self).__init__(**kwargs)

        word_index_path = get_valid('word_index_path', kwargs)
        self.word2num, self.num2word = self.generate_dict(word_index_path)
        self.class_num = len(self.word2num)
        self.max_len = get('max_len', kwargs, 64)
        self.mask_ratio = get('mask_ratio', kwargs, (8, 4))
        self.size = get('size', kwargs, (32, 256, 3))

        model = self.init_model(**kwargs)
        model = self.load_model(model=model, **kwargs)
        if self.use_gpu:
            model = model.cuda()
        self.model = model.eval()

    @staticmethod
    def generate_dict(word_index_path, encoding='utf-8'):
        word_index = open(word_index_path, 'r', encoding=encoding)
        word2num = dict()
        num2word = dict()
        word2num['SOS'] = 0
        num2word[0] = 'SOS'
        num = 1
        for line in word_index.readlines():
            line = line.replace(' ', '')
            line = line.replace('\n', '')
            for idx in range(len(line)):
                word = line[idx]
                if word not in word2num.keys():
                    word2num[word] = num
                    num2word[num] = word
                    num += 1
        word_index.close()
        return word2num, num2word

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

    def init_model(self, **kwargs):
        kwargs['class_num'] = self.class_num
        return super(PredictSarRecognition, self).init_model(model_factory=RecognitionModelFactory(), **kwargs)

    def load_model(self, **kwargs):
        return super(PredictSarRecognition, self).load_model(**kwargs)

    def pretreatment(self, **kwargs):
        eval_transforms = get('eval_transforms', kwargs, None)
        if eval_transforms is None:
            eval_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        img = get_valid('img', kwargs)
        img, valid_width = self.__padding(img)
        img = eval_transforms(img).unsqueeze(0)
        mask_h = int(self.size[0] / self.mask_ratio[0])
        mask_w = int(self.size[1] / self.mask_ratio[1])
        mask = np.zeros((1, 1, mask_h, mask_w))
        mask[:, :, :, :valid_width] = 1
        mask = torch.from_numpy(mask)

        return img, mask

    def predict(self, **kwargs):
        super(PredictSarRecognition, self).predict(**kwargs)
        img, mask = self.pretreatment(**kwargs)
        pred_str = ''
        with torch.no_grad():
            decoder_input = torch.zeros(1)
            if self.use_gpu:
                img = img.cuda()
                mask = mask.cuda()
                decoder_input = decoder_input.cuda()

            hidden, feature = self.model.encoder(image=img)
            hidden = hidden.permute(2, 0, 1)

            flag = True
            idx = 0
            while flag:
                output, hidden = self.model.decoder(input=decoder_input, hidden=hidden, feature=feature, mask=mask)
                _, topi = output.data.topk(1)
                decoder_input = topi.squeeze(1)

                if decoder_input == 0 or idx >= 64:
                    flag = False
                else:
                    cls = int(decoder_input)
                    pred = self.num2word[cls]
                    pred_str += pred

                idx += 1
        return pred_str

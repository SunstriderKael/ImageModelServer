from ModelHelper.Common.CommonUtils import get, get_valid
from abc import abstractmethod, ABCMeta
import torch
import os
from collections import OrderedDict


class AbstractPredict(metaclass=ABCMeta):
    def __init__(self, **kwargs):
        self.use_gpu = get('use_gpu', kwargs, True)
        if self.use_gpu:
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA is not availabel!')
        else:
            if torch.cuda.is_available():
                print('CUDA is availabel but not use it!')

        self.encoding = get('encoding', kwargs, 'utf-8')

    @abstractmethod
    def init_model(self, **kwargs):
        model_factory = get_valid('model_factory', kwargs)
        model = model_factory.get_model(**kwargs)
        return model

    @abstractmethod
    def load_model(self, **kwargs):
        model = get_valid('model', kwargs)
        checkpoint = get_valid('checkpoint', kwargs)
        assert os.path.exists(checkpoint)

        if self.use_gpu is False:
            ckpt = torch.load(checkpoint, map_location='cpu')
        else:
            ckpt = torch.load(checkpoint)
        new_state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        return model

    @abstractmethod
    def pretreatment(self, **kwargs):
        pass

    @abstractmethod
    def predict(self, **kwargs):
        pass

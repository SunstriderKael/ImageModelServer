from ModelHelper.Recognition.RecognitionModels.AbstractRecognitionModels import SarRecognitionModel, MultiDecoderSAR
from ModelHelper.Common.CommonUtils import get


class Resnet50SarRecognitionModel(SarRecognitionModel):
    def __init__(self, **kwargs):
        kwargs['encoder'] = get('encoder', kwargs, 'Encoder')
        kwargs['decoder'] = get('decoder', kwargs, 'Decoder')
        super(Resnet50SarRecognitionModel, self).__init__(**kwargs)

    def forward(self, **kwargs):
        return super(Resnet50SarRecognitionModel, self).forward(**kwargs)


class Resnet50MultiDecoderSarRecognitionModel(MultiDecoderSAR):
    def __init__(self, **kwargs):
        kwargs['encoder'] = get('encoder', kwargs, 'Encoder')
        kwargs['decoder'] = get('decoder', kwargs, 'Decoder')
        super(Resnet50MultiDecoderSarRecognitionModel, self).__init__(**kwargs)

    def forward(self, **kwargs):
        return super(Resnet50MultiDecoderSarRecognitionModel, self).forward(**kwargs)

from ModelHelper.Common.CommonModels.ModelFactory import AbstractModelFactory
from ModelHelper.Recognition import RecognitionModels
from ModelHelper.Recognition.RecognitionModels import EncoderModels, DecoderModels


class RecognitionModelFactory(AbstractModelFactory):
    def __init__(self, **kwargs):
        model_list = ['Resnet50SarRecognitionModel', 'Resnet50MultiDecoderSarRecognitionModel']
        super(RecognitionModelFactory, self).__init__(model_file=RecognitionModels, model_list=model_list, **kwargs)

    def get_model(self, **kwargs):
        return super(RecognitionModelFactory, self).get_model(**kwargs)


class EncoderModelFactory(AbstractModelFactory):
    def __init__(self, **kwargs):
        kwargs['model_file'] = EncoderModels
        super(EncoderModelFactory, self).__init__(**kwargs)

    def get_model(self, **kwargs):
        return super(EncoderModelFactory, self).get_model(**kwargs)


class DecoderModelFactory(AbstractModelFactory):
    def __init__(self, **kwargs):
        kwargs['model_file'] = DecoderModels
        super(DecoderModelFactory, self).__init__(**kwargs)

    def get_model(self, **kwargs):
        return super(DecoderModelFactory, self).get_model(**kwargs)

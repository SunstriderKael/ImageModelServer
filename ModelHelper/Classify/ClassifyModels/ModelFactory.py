from ModelHelper.Common.CommonModels.ModelFactory import AbstractModelFactory
from ModelHelper.Classify import ClassifyModels


class ClassifyModelFactory(AbstractModelFactory):
    def __init__(self, **kwargs):
        kwargs['model_file'] = ClassifyModels
        super(ClassifyModelFactory, self).__init__(**kwargs)

    def get_model(self, **kwargs):
        return super(ClassifyModelFactory, self).get_model(**kwargs)

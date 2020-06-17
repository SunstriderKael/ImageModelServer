from ModelHelper.Common import CommonModels
from ModelHelper.Common.CommonUtils import get_valid, get


class AbstractModelFactory:
    def __init__(self, **kwargs):
        self.model_file = get_valid('model_file', kwargs)
        self.model_list = get('model_list', kwargs)

    def get_model(self, **kwargs):
        model_name = get_valid('model_name', kwargs)
        if self.model_list is not None:
            try:
                assert model_name in self.model_list
            except:
                raise RuntimeError('model_name must in {}!'.format(self.model_list))
        model = getattr(self.model_file, model_name)
        return model(**kwargs)


class BackboneFactory(AbstractModelFactory):
    def __init__(self, **kwargs):
        kwargs['model_file'] = CommonModels
        super(BackboneFactory, self).__init__(**kwargs)

    def get_model(self, **kwargs):
        return super(BackboneFactory, self).get_model(**kwargs)

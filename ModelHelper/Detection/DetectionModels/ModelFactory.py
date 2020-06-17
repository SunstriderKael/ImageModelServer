from ModelHelper.Detection import DetectionModels
from ModelHelper.Common.CommonModels.ModelFactory import AbstractModelFactory


class DetectionModelFactory(AbstractModelFactory):
    def __init__(self, **kwargs):
        model_list = ['Fishnet99EastDetectionModel', 'Fishnet150EastDetectionModel', 'Resnet18EastDetectionModel',
                      'Resnet50EastDetectionModel', 'Resnet101EastDetectionModel', 'Resnet152EastDetectionModel',
                      'Fishnet99PseDetectionModel', 'Fishnet150PseDetectionModel', 'Resnet18PseDetectionMoel',
                      'Resnet50PseDetectionModel', 'Resnet101PseDetectionModel', 'Resnet152PseDetectionModel']
        super(DetectionModelFactory, self).__init__(model_file=DetectionModels, model_list=model_list, **kwargs)

    def get_model(self, **kwargs):
        return super(DetectionModelFactory, self).get_model(**kwargs)

from ModelHelper.Classify.ClassifyModels.AbstractModels import ResnetClassifyModel
from ModelHelper.Classify.ClassifyModels.AbstractModels import NTSClassifyModel


class Resnet18ClassifyModel(ResnetClassifyModel):
    def __init__(self, **kwargs):
        kwargs['backbone'] = 'resnet18'
        super(Resnet18ClassifyModel, self).__init__(fc_input_num=512, **kwargs)

    def forward(self, **kwargs):
        return super(Resnet18ClassifyModel, self).forward(**kwargs)


class Resnet34ClassifyModel(ResnetClassifyModel):
    def __init__(self, **kwargs):
        kwargs['backbone'] = 'resnet34'
        super(Resnet34ClassifyModel, self).__init__(fc_input_num=512, **kwargs)

    def forward(self, **kwargs):
        return super(Resnet34ClassifyModel, self).forward(**kwargs)


class Resnet50ClassifyModel(ResnetClassifyModel):
    def __init__(self, **kwargs):
        kwargs['backbone'] = 'resnet50'
        super(Resnet50ClassifyModel, self).__init__(fc_input_num=512, **kwargs)

    def forward(self, **kwargs):
        return super(Resnet50ClassifyModel, self).forward(**kwargs)


class Resnet101ClassifyModel(ResnetClassifyModel):
    def __init__(self, **kwargs):
        kwargs['backbone'] = 'resnet101'
        super(Resnet101ClassifyModel, self).__init__(fc_input_num=512, **kwargs)

    def forward(self, **kwargs):
        return super(Resnet101ClassifyModel, self).forward(**kwargs)


class Resnet152ClassifyModel(ResnetClassifyModel):
    def __init__(self, **kwargs):
        kwargs['backbone'] = 'resnet152'
        super(Resnet152ClassifyModel, self).__init__(fc_input_num=512, **kwargs)

    def forward(self, **kwargs):
        return super(Resnet152ClassifyModel, self).forward(**kwargs)


class Resnet18NTSClassifyModel(NTSClassifyModel):
    def __init__(self, **kwargs):
        kwargs['backbone'] = 'resnet18'
        super(Resnet18NTSClassifyModel, self).__init__(nts_fc_ratio=1, **kwargs)

    def forward(self, **kwargs):
        return super(Resnet18NTSClassifyModel, self).forward(**kwargs)


class Resnet34NTSClassifyModel(NTSClassifyModel):
    def __init__(self, **kwargs):
        kwargs['backbone'] = 'resnet34'
        super(Resnet34NTSClassifyModel, self).__init__(nts_fc_ratio=1, **kwargs)

    def forward(self, **kwargs):
        return super(Resnet34NTSClassifyModel, self).forward(**kwargs)


class Resnet50NTSClassifyModel(NTSClassifyModel):
    def __init__(self, **kwargs):
        kwargs['backbone'] = 'resnet50'
        super(Resnet50NTSClassifyModel, self).__init__(nts_fc_ratio=4, **kwargs)

    def forward(self, **kwargs):
        return super(Resnet50NTSClassifyModel, self).forward(**kwargs)


class Resnet101NTSClassifyModel(NTSClassifyModel):
    def __init__(self, **kwargs):
        kwargs['backbone'] = 'resnet101'
        super(Resnet101NTSClassifyModel, self).__init__(nts_fc_ratio=4, **kwargs)

    def forward(self, **kwargs):
        return super(Resnet101NTSClassifyModel, self).forward(**kwargs)


class Resnet152NTSClassifyModel(NTSClassifyModel):
    def __init__(self, **kwargs):
        kwargs['backbone'] = 'resnet152'
        super(Resnet152NTSClassifyModel, self).__init__(nts_fc_ratio=4, **kwargs)

    def forward(self, **kwargs):
        return super(Resnet152NTSClassifyModel, self).forward(**kwargs)

from ModelHelper.Common.CommonModels.Fishnet import fish
from ModelHelper.Common.CommonModels.Resnet import ResNet, BasicBlock, Bottleneck
from ModelHelper.Common.CommonUtils import get
from torch.utils import model_zoo
import torch


def fishnet99(**kwargs):
    net_cfg = {
        #  input size:   [224, 56, 28,  14  |  7,   7,  14,  28 | 56,   28,  14]
        # output size:   [56,  28, 14,   7  |  7,  14,  28,  56 | 28,   14,   7]
        #                  |    |    |   |     |    |    |    |    |     |    |
        'network_planes': [64, 128, 256, 512, 512, 512, 384, 256, 320, 832, 1600],
        'num_res_blks': [2, 2, 6, 2, 1, 1, 1, 1, 2, 2],
        'num_trans_blks': [1, 1, 1, 1, 1, 4],
        'num_cls': 1000,
        'num_down_sample': 3,
        'num_up_sample': 3,
    }
    cfg = {**net_cfg, **kwargs}
    return fish(**cfg)


def fishnet150(**kwargs):
    net_cfg = {
        #  input size:   [224, 56, 28,  14  |  7,   7,  14,  28 | 56,   28,  14]
        # output size:   [56,  28, 14,   7  |  7,  14,  28,  56 | 28,   14,   7]
        #                  |    |    |   |     |    |    |    |    |     |    |
        'network_planes': [64, 128, 256, 512, 512, 512, 384, 256, 320, 832, 1600],
        'num_res_blks': [2, 4, 8, 4, 2, 2, 2, 2, 2, 4],
        'num_trans_blks': [2, 2, 2, 2, 2, 4],
        'num_cls': 1000,
        'num_down_sample': 3,
        'num_up_sample': 3,
    }
    cfg = {**net_cfg, **kwargs}
    return fish(**cfg)


def fishnet201(**kwargs):
    net_cfg = {
        #  input size:   [224, 56, 28,  14  |  7,   7,  14,  28 | 56,   28,  14]
        # output size:   [56,  28, 14,   7  |  7,  14,  28,  56 | 28,   14,   7]
        #                  |    |    |   |     |    |    |    |    |     |    |
        'network_planes': [64, 128, 256, 512, 512, 512, 384, 256, 320, 832, 1600],
        'num_res_blks': [3, 4, 12, 4, 2, 2, 2, 2, 3, 10],
        'num_trans_blks': [2, 2, 2, 2, 2, 9],
        'num_cls': 1000,
        'num_down_sample': 3,
        'num_up_sample': 3,
    }
    cfg = {**net_cfg, **kwargs}
    return fish(**cfg)


resnet_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def resnet18(**kwargs):
    pretrained = get('pretrained_backbone', kwargs, False)
    backbone_path = get('backbone_path', kwargs, None)
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained is True:
        if backbone_path is not None:
            ckpt = torch.load(backbone_path)
        else:
            url = resnet_urls['resnet18']
            ckpt = model_zoo.load_url(url)
        model.load_state_dict(ckpt)

    return model


def resnet34(**kwargs):
    pretrained = get('pretrained_backbone', kwargs, False)
    backbone_path = get('backbone_path', kwargs, None)
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained is True:
        if backbone_path is not None:
            ckpt = torch.load(backbone_path)
        else:
            url = resnet_urls['resnet34']
            ckpt = model_zoo.load_url(url)
        model.load_state_dict(ckpt)

    return model


def resnet50(**kwargs):
    pretrained = get('pretrained_backbone', kwargs, False)
    backbone_path = get('backbone_path', kwargs, None)
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained is True:
        if backbone_path is not None:
            ckpt = torch.load(backbone_path)
        else:
            url = resnet_urls['resnet50']
            ckpt = model_zoo.load_url(url)
        model.load_state_dict(ckpt)

    return model


def resnet101(**kwargs):
    pretrained = get('pretrained_backbone', kwargs, False)
    backbone_path = get('backbone_path', kwargs, None)
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained is True:
        if backbone_path is not None:
            ckpt = torch.load(backbone_path)
        else:
            url = resnet_urls['resnet101']
            ckpt = model_zoo.load_url(url)
        model.load_state_dict(ckpt)

    return model


def resnet152(**kwargs):
    pretrained = get('pretrained_backbone', kwargs, False)
    backbone_path = get('backbone_path', kwargs, None)
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained is True:
        if backbone_path is not None:
            ckpt = torch.load(backbone_path)
        else:
            url = resnet_urls['resnet152']
            ckpt = model_zoo.load_url(url)
        model.load_state_dict(ckpt)

    return model

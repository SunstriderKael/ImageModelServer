import torch.nn as nn
import torch
import math
from ModelHelper.Common.CommonUtils import get, get_valid
from ModelHelper.Common.CommonModels.ModelFactory import BackboneFactory
import torch.nn.functional as F


class AbstractDetectionModel(nn.Module):
    def __init__(self, **kwargs):
        super(AbstractDetectionModel, self).__init__()
        backbone = get_valid('backbone', kwargs)
        backbone_factory = BackboneFactory()
        kwargs['model_name'] = backbone
        self.backbone = backbone_factory.get_model(**kwargs)

    def forward(self, image):
        pass


class EastDetectionModel(AbstractDetectionModel):
    def __init__(self, **kwargs):
        super(EastDetectionModel, self).__init__(**kwargs)
        channels = get_valid('channels', kwargs)
        self.detection_size = get('detection_size', kwargs, 768)

        self.conv1_in_channels = channels[0]
        self.conv3_in_channels = channels[1]
        self.conv5_in_channels = channels[2]

        self.conv1 = nn.Conv2d(self.conv1_in_channels, 128, 1)
        self.bn1 = nn.BatchNorm2d(128, momentum=0.9)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128, momentum=0.9)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(self.conv3_in_channels, 64, 1)
        self.bn3 = nn.BatchNorm2d(64, momentum=0.9)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64, momentum=0.9)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(self.conv5_in_channels, 64, 1)
        self.bn5 = nn.BatchNorm2d(64, momentum=0.9)
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv2d(64, 32, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(32, momentum=0.9)
        self.relu6 = nn.ReLU()

        self.conv7 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(32, momentum=0.9)
        self.relu7 = nn.ReLU()

        self.conv8 = nn.Conv2d(32, 1, 1)
        self.sigmoid = nn.Sigmoid()
        self.conv9 = nn.Conv2d(32, 4, 1)
        self.conv10 = nn.Conv2d(32, 1, 1)

        self.uppool1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.uppool2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.uppool3 = nn.Upsample(scale_factor=2, mode='bilinear')

    def match_shape(self, feature, g):
        if feature.shape[2] != g.shape[2]:
            if feature.shape[2] > g.shape[2]:
                min_size = g.shape[2]
                feature = feature[:, :, 0:min_size, :]
            else:
                min_size = feature.shape[2]
                g = g[:, :, 0:min_size, :]

        if feature.shape[3] != g.shape[3]:
            if feature.shape[3] > g.shape[3]:
                min_size = g.shape[3]
                feature = feature[:, :, :, 0:min_size]
            else:
                min_size = feature.shape[3]
                g = g[:, :, :, 0:min_size]
        return feature, g

    def forward(self, image):
        super(EastDetectionModel, self).forward(image)
        feature_list = self.backbone(image)

        h = feature_list[3]
        g = self.uppool1(h)
        feature_2 = feature_list[2]
        if feature_2.shape[2] != g.shape[2] or feature_2.shape[3] != g.shape[3]:
            feature_2, g = self.match_shape(feature_2, g)
        c = self.conv1(torch.cat((g, feature_2), 1))
        c = self.bn1(c)
        c = self.relu1(c)

        h = self.conv2(c)
        h = self.bn2(h)
        h = self.relu2(h)

        g = self.uppool2(h)

        feature_1 = feature_list[1]
        if feature_1.shape[2] != g.shape[2] or feature_1.shape[3] != g.shape[3]:
            feature_1, g = self.match_shape(feature_1, g)

        c = self.conv3(torch.cat((g, feature_1), 1))
        c = self.bn3(c)
        c = self.relu3(c)

        h = self.conv4(c)  # bs 64 w/8 h/8
        h = self.bn4(h)
        h = self.relu4(h)
        g = self.uppool3(h)  # bs 64 w/4 h/4

        feature_0 = feature_list[0]
        if feature_0.shape[2] != g.shape[2] or feature_0.shape[3] != g.shape[3]:
            feature_0, g = self.match_shape(feature_0, g)

        c = self.conv5(torch.cat((g, feature_0), 1))
        c = self.bn5(c)
        c = self.relu5(c)

        h = self.conv6(c)  # bs 32 w/4 h/4
        h = self.bn6(h)
        h = self.relu6(h)
        g = self.conv7(h)  # bs 32 w/4 h/4
        g = self.bn7(g)
        g = self.relu7(g)

        F_score = self.conv8(g)  # bs 1 w/4 h/4
        F_score = self.sigmoid(F_score)
        geo_map = self.conv9(g)
        geo_map = self.sigmoid(geo_map) * self.detection_size
        angle_map = self.conv10(g)
        angle_map = self.sigmoid(angle_map)
        angle_map = (angle_map - 0.5) * math.pi / 2

        F_geometry = torch.cat((geo_map, angle_map), 1)  # bs 5 w/4 w/4
        return F_score, F_geometry


class PseDetectionModel(AbstractDetectionModel):
    def __init__(self, **kwargs):
        super(PseDetectionModel, self).__init__(**kwargs)
        n = get('n', kwargs, 6)
        scale = get('scale', kwargs, 1)
        channels = get_valid('channels', kwargs)
        self.scale = scale
        conv_out = 256

        # Top layer
        self.toplayer = nn.Conv2d(channels[0], conv_out, kernel_size=1, stride=1, padding=0)  # Reduce channels
        # Lateral layers
        self.latlayer1 = nn.Conv2d(channels[1], conv_out, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(channels[2], conv_out, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(channels[3], conv_out, kernel_size=1, stride=1, padding=0)

        # Smooth layers
        self.smooth1 = nn.Conv2d(conv_out, conv_out, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(conv_out, conv_out, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(conv_out, conv_out, kernel_size=3, stride=1, padding=1)

        self.conv = nn.Sequential(
            nn.Conv2d(conv_out * 4, conv_out, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(conv_out),
            nn.ReLU(inplace=True)
        )
        self.out_conv = nn.Conv2d(conv_out, n, kernel_size=1, stride=1)

    def forward(self, image):
        _, _, H, W = image.size()
        feature_list = self.backbone(image)
        c2 = feature_list[3]
        c3 = feature_list[2]
        c4 = feature_list[1]
        c5 = feature_list[0]
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self.__upsample_add(p5, self.latlayer1(c4))
        p3 = self.__upsample_add(p4, self.latlayer2(c3))
        p2 = self.__upsample_add(p3, self.latlayer3(c2))
        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)

        x = self.__upsample_cat(p2, p3, p4, p5)
        x = self.conv(x)
        x = self.out_conv(x)

        if self.train:
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        else:
            x = F.interpolate(x, size=(H // self.scale, W // self.scale), mode='bilinear', align_corners=True)
        return x

    def __upsample_add(self, x, y):
        return F.interpolate(x, size=y.size()[2:], mode='bilinear', align_corners=False) + y

    def __upsample_cat(self, p2, p3, p4, p5):
        h, w = p2.size()[2:]
        p3 = F.interpolate(p3, size=(h, w), mode='bilinear', align_corners=False)
        p4 = F.interpolate(p4, size=(h, w), mode='bilinear', align_corners=False)
        p5 = F.interpolate(p5, size=(h, w), mode='bilinear', align_corners=False)
        return torch.cat([p2, p3, p4, p5], dim=1)

from ModelHelper.Common.CommonUtils import get, get_valid
import torch.nn.functional as F

import torch.nn as nn
from ModelHelper.Common.CommonUtils import get


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.9)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.9)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, **kwargs):
        super(ResNet, self).__init__()
        self.inplanes = 128

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool_vertical = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        self.layer1 = self._make_layer(block, 256, layers[0], stride=1)

        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)

        self.layer2 = self._make_layer(block, 256, layers[1], stride=1)

        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(256)

        self.layer3 = self._make_layer(block, 512, layers[2], stride=1)

        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(512)

        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)

        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(512)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.maxpool(x)
        x = self.layer2(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.maxpool_vertical(x)
        x = self.layer3(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        x = self.layer4(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)

        return x


def resnet50(**kwargs):
    model = ResNet(BasicBlock, [1, 2, 5, 3], **kwargs)
    return model


class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__()

        self.backbone = resnet50(**kwargs)
        nh = get('nh', kwargs, 512)
        self.rnn = nn.GRU(512, nh, num_layers=2, bidirectional=False)

    def forward(self, **kwargs):
        image = get_valid('image', kwargs)
        feature = self.backbone(image)
        inputs = F.max_pool2d(feature, kernel_size=(4, 1), stride=(4, 1), ceil_mode=True)

        inputs = inputs.squeeze(2)
        inputs = inputs.permute(2, 0, 1)

        _, hidden = self.rnn(inputs)
        hidden = hidden.permute(1, 2, 0)
        return hidden, feature


class EncoderX16(nn.Module):
    def __init__(self, **kwargs):
        super(EncoderX16, self).__init__()

        self.backbone = resnet50(**kwargs)
        nh = get('nh', kwargs, 512)
        self.rnn = nn.GRU(512, nh, num_layers=2, bidirectional=False)

    def forward(self, **kwargs):
        image = get_valid('image', kwargs)
        feature = self.backbone(image)
        feature = F.max_pool2d(feature, kernel_size=(4, 4), stride=(4, 4), ceil_mode=True)

        inputs = feature.squeeze(2)
        inputs = inputs.permute(2, 0, 1)

        _, hidden = self.rnn(inputs)
        hidden = hidden.permute(1, 2, 0)
        return hidden, feature

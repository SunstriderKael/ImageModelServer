from ModelHelper.Common.CommonUtils import get, get_valid
import torch.nn as nn
import torch


class Decoder(nn.Module):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__()
        self.hidden_size = get('hidden_size', kwargs, 512)
        self.class_num = get_valid('class_num', kwargs)
        self.embedding = nn.Embedding(self.class_num, self.hidden_size)
        self.dropout_p = get('dropout_p', kwargs, 0.1)
        self.f_channel = get('f_channel', kwargs, 512)
        # print('**** decoder dropout ratio: {} ****'.format(self.dropout_p))
        self.dropout = nn.Dropout(self.dropout_p)
        self.rnn = nn.GRU(self.hidden_size, self.hidden_size, num_layers=2)
        self.out = nn.Linear(self.hidden_size + self.f_channel, self.class_num)
        self.conv1 = nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(512, 1, kernel_size=1, stride=1, bias=False)
        # self.feature_conv = nn.Conv2d(512, self.f_channel, kernel_size=3, stride=1, padding=1, bias=False)
        # self.relu = nn.ReLU(inplace=True)
        # self.bn = nn.BatchNorm2d(self.f_channel)

    def forward(self, **kwargs):
        input = get_valid('input', kwargs)
        hidden = get_valid('hidden', kwargs)
        feature = get_valid('feature', kwargs)
        mask = get_valid('mask', kwargs)

        input = input.long()
        input = self.embedding(input)
        input = self.dropout(input)
        input = input.unsqueeze(0)

        _, hidden = self.rnn(input, hidden)
        hidden_tmp = hidden[1:].permute(1, 2, 0).unsqueeze(3)
        hidden_tmp = self.conv1(hidden_tmp)
        hidden_tmp = hidden_tmp.expand_as(feature)

        encode_conv = self.conv2(feature)
        encode_conv = self.conv3(torch.tanh(encode_conv + hidden_tmp)).view(encode_conv.shape[0], 1, -1)
        mask = mask.view(mask.shape[0], mask.shape[1], -1)

        w = self.mask_softmax(encode_conv, dim=2, mask=mask)
        # feature = self.feature_conv(feature)
        # feature = self.bn(feature)
        # feature = self.relu(feature)
        feature = feature.view(feature.shape[0], feature.shape[1], -1)
        c = torch.sum(feature * w, 2)
        ouput = torch.cat([hidden[1], c], 1)
        ouput = self.out(ouput)
        return ouput, hidden

    @staticmethod
    def mask_softmax(input, dim, mask):
        input_max = torch.max(input, dim=dim, keepdim=True)[0]
        input_exp = torch.exp(input - input_max)

        input_exp = input_exp * mask.float()
        input_softmax = input_exp / torch.sum(input_exp, dim=dim, keepdim=True) + 0.000001
        return input_softmax

from ModelHelper.Common.CommonUtils import get_valid, get
from ModelHelper.Recognition.RecognitionModels.ModelFactory import EncoderModelFactory, DecoderModelFactory
import torch.nn as nn
import random
import torch


class AbstractEncoderDecoderModel(nn.Module):
    def __init__(self, **kwargs):
        super(AbstractEncoderDecoderModel, self).__init__()
        encoder = get_valid('encoder', kwargs)
        decoder = get_valid('decoder', kwargs)
        model_name = get_valid('model_name', kwargs)

        encoder_factory = EncoderModelFactory()
        kwargs['model_name'] = encoder
        self.encoder = encoder_factory.get_model(**kwargs)

        decoder_factory = DecoderModelFactory()
        kwargs['model_name'] = decoder
        self.decoder = decoder_factory.get_model(**kwargs)

        kwargs['model_name'] = model_name

    def forward(self, **kwargs):
        pass


class MultiDecoderSAR(nn.Module):
    def __init__(self, **kwargs):
        super(MultiDecoderSAR, self).__init__()
        encoder_name = get_valid('encoder', kwargs)
        decoder_name = get_valid('decoder', kwargs)

        self.decoder_num = get_valid('decoder_num', kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, self.decoder_num)

        encoder_factory = EncoderModelFactory()
        kwargs['model_name'] = encoder_name
        self.encoder = encoder_factory.get_model(**kwargs)
        self.class_num_list = get_valid('class_num_list', kwargs)

        self.decoder_list = list()
        for idx in range(self.decoder_num):
            decoder_factory = DecoderModelFactory()
            kwargs['model_name'] = decoder_name
            class_num = self.class_num_list[idx]
            decoder = decoder_factory.get_model(class_num=class_num, **kwargs)
            if torch.cuda.is_available():
                decoder.cuda()
            self.decoder_list.append(decoder)

    def forward(self, **kwargs):
        type = get_valid('type', kwargs)
        if type == 'classify':
            image = get_valid('image', kwargs)
            hidden, feature = self.encoder(image=image)
            hidden = hidden.permute(2, 0, 1)

            x = self.avgpool(feature)
            x = x.view(x.size(0), -1)
            x = nn.Dropout(p=0.5)(x)
            x = self.fc(x)
            return x
        elif type == 'sar':
            image = get_valid('image', kwargs)
            target_variable = get_valid('target', kwargs)
            mask = get_valid('mask', kwargs)
            teacher_forcing_ratio = get('teacher_forcing_ratio', kwargs, 1)
            cls_label = get_valid('cls_label', kwargs)

            hidden, feature = self.encoder(image=image)
            hidden = hidden.permute(2, 0, 1)

            x = self.avgpool(feature)
            x = x.view(x.size(0), -1)
            x = nn.Dropout(p=0.5)(x)
            x = self.fc(x)

            decoder = self.decoder_list[cls_label]

            decoder_input = target_variable[:, 0]
            output_list = list()

            for di in range(1, target_variable.shape[1]):
                output, hidden = decoder(input=decoder_input, hidden=hidden, feature=feature, mask=mask)
                output_list.append(output.unsqueeze(1))

                teacher_forcing = random.random() < teacher_forcing_ratio
                if teacher_forcing:
                    decoder_input = target_variable[:, di]

                else:
                    _, topi = output.data.topk(1)
                    decoder_input = topi.squeeze(1)

            output_list = torch.cat(output_list, 1)
            return output_list, x


class SarRecognitionModel(AbstractEncoderDecoderModel):
    def __int__(self, **kwargs):
        super(SarRecognitionModel, self).__init__(**kwargs)
        pass

    def forward(self, **kwargs):
        super(SarRecognitionModel, self).forward(**kwargs)
        image = get_valid('image', kwargs)
        target_variable = get_valid('target', kwargs)
        mask = get_valid('mask', kwargs)
        teacher_forcing_ratio = get('teacher_forcing_ratio', kwargs, 1)

        hidden, feature = self.encoder(image=image)
        hidden = hidden.permute(2, 0, 1)

        decoder_input = target_variable[:, 0]
        output_list = list()

        for di in range(1, target_variable.shape[1]):
            output, hidden = self.decoder(input=decoder_input, hidden=hidden, feature=feature, mask=mask)
            output_list.append(output.unsqueeze(1))

            teacher_forcing = random.random() < teacher_forcing_ratio
            if teacher_forcing:
                decoder_input = target_variable[:, di]

            else:
                _, topi = output.data.topk(1)
                decoder_input = topi.squeeze(1)

        output_list = torch.cat(output_list, 1)
        return output_list

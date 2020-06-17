from ModelHelper.Common.CommonUtils import get, get_valid
from ModelHelper.Common.CommonModels.ModelFactory import BackboneFactory
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AbstractClassifyModel(nn.Module):
    def __init__(self, **kwargs):
        super(AbstractClassifyModel, self).__init__()
        backbone = get_valid('backbone', kwargs)
        backbone_factory = BackboneFactory()
        kwargs['model_name'] = backbone
        self.backbone = backbone_factory.get_model(**kwargs)

    def forward(self, **kwargs):
        pass


# NTS Net
class ProposalNet(nn.Module):
    def __init__(self, **kwargs):
        super(ProposalNet, self).__init__()
        self.nts_fc_ratio = get_valid("nts_fc_ratio", kwargs)
        self.down1 = nn.Conv2d(512*self.nts_fc_ratio, 128, 3, 1, 1)
        self.down2 = nn.Conv2d(128, 128, 3, 2, 1)
        self.down3 = nn.Conv2d(128, 128, 3, 2, 1)
        self.ReLU = nn.ReLU()
        self.tidy1 = nn.Conv2d(128, 6, 1, 1, 0)
        self.tidy2 = nn.Conv2d(128, 6, 1, 1, 0)
        self.tidy3 = nn.Conv2d(128, 9, 1, 1, 0)

    def forward(self, x):
        batch_size = x.size(0)
        d1 = self.ReLU(self.down1(x))
        d2 = self.ReLU(self.down2(d1))
        d3 = self.ReLU(self.down3(d2))
        t1 = self.tidy1(d1).view(batch_size, -1)
        t2 = self.tidy2(d2).view(batch_size, -1)
        t3 = self.tidy3(d3).view(batch_size, -1)
        return torch.cat((t1, t2, t3), dim=1)


class NTSClassifyModel(AbstractClassifyModel):
    def __init__(self, **kwargs):
        kwargs['get_layer4_feature'] = True
        kwargs['get_fc_feature'] = True
        kwargs['get_fc_output'] = True
        self.nts_fc_ratio = get_valid("nts_fc_ratio", kwargs)
        self.use_gpu = get('use_gpu', kwargs, True)
        self.class_num = get_valid('class_num', kwargs)
        super(NTSClassifyModel, self).__init__(**kwargs)

        self.top_n = get('top_n', kwargs, 6)
        self.crop_size = get('crop_size', kwargs, (448, 448))
        self.cat_num = get('cat_num', kwargs, 4)
        self._default_anchors_setting = (
            dict(layer='p3', stride=32, size=48, scale=[2 ** (1. / 3.), 2 ** (2. / 3.)], aspect_ratio=[0.667, 1, 1.5]),
            dict(layer='p4', stride=64, size=96, scale=[2 ** (1. / 3.), 2 ** (2. / 3.)], aspect_ratio=[0.667, 1, 1.5]),
            dict(layer='p5', stride=128, size=192, scale=[1, 2 ** (1. / 3.), 2 ** (2. / 3.)],
                 aspect_ratio=[0.667, 1, 1.5]),
        )

        self.backbone.avgpool = nn.AdaptiveAvgPool2d(1)
        self.backbone.fc = nn.Linear(512*self.nts_fc_ratio, self.class_num)
        self.proposal_net = ProposalNet(**kwargs)
        self.concat_net = nn.Linear(512*self.nts_fc_ratio * (self.cat_num + 1), self.class_num)
        self.partcls_net = nn.Linear(512 * self.nts_fc_ratio, self.class_num)
        _, edge_anchors, _ = self.generate_default_anchor_maps()
        self.pad_side = 224
        self.edge_anchors = (edge_anchors + 224).astype(np.int)

    def generate_default_anchor_maps(self, anchors_setting=None):
        """
        generate default anchor

        :param anchors_setting: all informations of anchors
        :param input_shape: shape of input images, e.g. (h, w)
        :return: center_anchors: # anchors * 4 (oy, ox, h, w)
                 edge_anchors: # anchors * 4 (y0, x0, y1, x1)
                 anchor_area: # anchors * 1 (area)
        """
        input_shape = self.crop_size
        if anchors_setting is None:
            anchors_setting = self._default_anchors_setting

        center_anchors = np.zeros((0, 4), dtype=np.float32)
        edge_anchors = np.zeros((0, 4), dtype=np.float32)
        anchor_areas = np.zeros((0,), dtype=np.float32)
        input_shape = np.array(input_shape, dtype=int)

        for anchor_info in anchors_setting:

            stride = anchor_info['stride']
            size = anchor_info['size']
            scales = anchor_info['scale']
            aspect_ratios = anchor_info['aspect_ratio']

            output_map_shape = np.ceil(input_shape.astype(np.float32) / stride)
            output_map_shape = output_map_shape.astype(np.int)
            output_shape = tuple(output_map_shape) + (4,)
            ostart = stride / 2.
            oy = np.arange(ostart, ostart + stride * output_shape[0], stride)
            oy = oy.reshape(output_shape[0], 1)
            ox = np.arange(ostart, ostart + stride * output_shape[1], stride)
            ox = ox.reshape(1, output_shape[1])
            center_anchor_map_template = np.zeros(output_shape, dtype=np.float32)
            center_anchor_map_template[:, :, 0] = oy
            center_anchor_map_template[:, :, 1] = ox
            for scale in scales:
                for aspect_ratio in aspect_ratios:
                    center_anchor_map = center_anchor_map_template.copy()
                    center_anchor_map[:, :, 2] = size * scale / float(aspect_ratio) ** 0.5
                    center_anchor_map[:, :, 3] = size * scale * float(aspect_ratio) ** 0.5

                    edge_anchor_map = np.concatenate((center_anchor_map[..., :2] - center_anchor_map[..., 2:4] / 2.,
                                                      center_anchor_map[..., :2] + center_anchor_map[..., 2:4] / 2.),
                                                     axis=-1)
                    anchor_area_map = center_anchor_map[..., 2] * center_anchor_map[..., 3]
                    center_anchors = np.concatenate((center_anchors, center_anchor_map.reshape(-1, 4)))
                    edge_anchors = np.concatenate((edge_anchors, edge_anchor_map.reshape(-1, 4)))
                    anchor_areas = np.concatenate((anchor_areas, anchor_area_map.reshape(-1)))

        return center_anchors, edge_anchors, anchor_areas

    @staticmethod
    def hard_nms(cdds, topn=10, iou_thresh=0.25):
        if not (type(cdds).__module__ == 'numpy' and len(cdds.shape) == 2 and cdds.shape[1] >= 5):
            raise TypeError('edge_box_map should be N * 5+ ndarray')

        cdds = cdds.copy()
        indices = np.argsort(cdds[:, 0])
        cdds = cdds[indices]
        cdd_results = []

        res = cdds

        while res.any():
            cdd = res[-1]
            cdd_results.append(cdd)
            if len(cdd_results) == topn:
                return np.array(cdd_results)
            res = res[:-1]

            start_max = np.maximum(res[:, 1:3], cdd[1:3])
            end_min = np.minimum(res[:, 3:5], cdd[3:5])
            lengths = end_min - start_max
            intersec_map = lengths[:, 0] * lengths[:, 1]
            intersec_map[np.logical_or(lengths[:, 0] < 0, lengths[:, 1] < 0)] = 0
            iou_map_cur = intersec_map / ((res[:, 3] - res[:, 1]) * (res[:, 4] - res[:, 2]) + (cdd[3] - cdd[1]) * (
                    cdd[4] - cdd[2]) - intersec_map)
            res = res[iou_map_cur < iou_thresh]

        return np.array(cdd_results)

    def forward(self, **kwargs):
        super(NTSClassifyModel, self).forward(**kwargs)
        image = get_valid('image', kwargs)

        feature_list = self.backbone(image)
        resnet_out = feature_list[2]
        rpn_feature = feature_list[0]
        feature = feature_list[1]

        x_pad = F.pad(image, (self.pad_side, self.pad_side, self.pad_side, self.pad_side), mode='constant', value=0)
        batch = image.size(0)
        # we will reshape rpn to shape: batch * nb_anchor
        rpn_score = self.proposal_net(rpn_feature.detach())
        all_cdds = [
            np.concatenate((x.reshape(-1, 1), self.edge_anchors.copy(), np.arange(0, len(x)).reshape(-1, 1)), axis=1)
            for x in rpn_score.data.cpu().numpy()]
        top_n_cdds = [self.hard_nms(x, topn=self.top_n, iou_thresh=0.25) for x in all_cdds]
        top_n_cdds = np.array(top_n_cdds)
        top_n_index = top_n_cdds[:, :, -1].astype(np.int)
        if self.use_gpu:
            top_n_index = torch.from_numpy(top_n_index).cuda()
        else:
            top_n_index = torch.from_numpy(top_n_index)
        top_n_prob = torch.gather(rpn_score, dim=1, index=top_n_index.long())
        if self.use_gpu:
            part_imgs = torch.zeros([batch, self.top_n, 3, 224, 224]).cuda()
        else:
            part_imgs = torch.zeros([batch, self.top_n, 3, 224, 224])
        for i in range(batch):
            for j in range(self.top_n):
                [y0, x0, y1, x1] = top_n_cdds[i][j, 1:5].astype(np.int)
                part_imgs[i:i + 1, j] = F.interpolate(x_pad[i:i + 1, :, y0:y1, x0:x1], size=(224, 224), mode='bilinear',
                                                      align_corners=True)
        part_imgs = part_imgs.view(batch * self.top_n, 3, 224, 224)
        feature_list2 = self.backbone(part_imgs.detach())
        part_features = feature_list2[1]
        part_feature = part_features.view(batch, self.top_n, -1)
        part_feature = part_feature[:, :self.cat_num, ...].contiguous()
        part_feature = part_feature.view(batch, -1)
        concat_out = torch.cat([part_feature, feature], dim=1)
        concat_logits = self.concat_net(concat_out)
        raw_logits = resnet_out
        part_logits = self.partcls_net(part_features).view(batch, self.top_n, -1)
        return [raw_logits, concat_logits, part_logits, top_n_index, top_n_prob]

    @staticmethod
    def list_loss(logits, targets):
        temp = F.log_softmax(logits, -1)
        loss = [-temp[i][targets[i].item()] for i in range(logits.size(0))]
        return torch.stack(loss)

    @staticmethod
    def ranking_loss(score, targets, proposal_num, **kwargs):
        use_gpu = get('use_gpu', kwargs, True)
        if use_gpu:
            loss = Variable(torch.zeros(1).cuda())
        else:
            loss = Variable(torch.zeros(1))
        batch_size = score.size(0)
        for i in range(proposal_num):
            if use_gpu:
                targets_p = (targets > targets[:, i].unsqueeze(1)).type(torch.cuda.FloatTensor)
            else:
                targets_p = (targets > targets[:, i].unsqueeze(1)).type(torch.FloatTensor)
            pivot = score[:, i].unsqueeze(1)
            loss_p = (1 - pivot + score) * targets_p
            loss_p = torch.sum(F.relu(loss_p))
            loss += loss_p
        return loss / batch_size


class ResnetClassifyModel(AbstractClassifyModel):
    def __init__(self, **kwargs):
        kwargs['get_fc_feature'] = True
        super(ResnetClassifyModel, self).__init__(**kwargs)
        self.class_num = get_valid('class_num', kwargs)
        fc_input_num = get_valid('fc_input_num', kwargs)
        self.fc = nn.Linear(fc_input_num, self.class_num)

    def forward(self, **kwargs):
        super(ResnetClassifyModel, self).forward(**kwargs)
        image = get_valid('image', kwargs)
        feature_list = self.backbone(image)
        output = self.fc(feature_list[0])
        return output

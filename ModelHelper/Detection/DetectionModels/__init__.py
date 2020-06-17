from ModelHelper.Detection.DetectionModels.AbstractModels import EastDetectionModel, PseDetectionModel
from ModelHelper.Common.CommonUtils import get
import numpy as np
import torch.nn as nn
import torch


class Fishnet99EastDetectionModel(EastDetectionModel):
    def __init__(self, **kwargs):
        kwargs['backbone'] = 'fishnet99'
        kwargs['channels'] = [2656, 960, 384]
        super(Fishnet99EastDetectionModel, self).__init__(**kwargs)

    def forward(self, image):
        return super(Fishnet99EastDetectionModel, self).forward(image)


class Fishnet150EastDetectionModel(EastDetectionModel):
    def __init__(self, **kwargs):
        kwargs['backbone'] = 'fishnet150'
        kwargs['channels'] = [2656, 960, 384]
        super(Fishnet150EastDetectionModel, self).__init__(**kwargs)

    def forward(self, image):
        return super(Fishnet150EastDetectionModel, self).forward(image)


class Resnet18EastDetectionModel(EastDetectionModel):
    def __init__(self, **kwargs):
        kwargs['backbone'] = 'resnet18'
        kwargs['channels'] = [768, 256, 128]
        kwargs['get_layer1_feature'] = True
        kwargs['get_layer2_feature'] = True
        kwargs['get_layer3_feature'] = True
        kwargs['get_layer4_feature'] = True
        super(Resnet18EastDetectionModel, self).__init__(**kwargs)

    def forward(self, image):
        return super(Resnet18EastDetectionModel, self).forward(image)


class Resnet50EastDetectionModel(EastDetectionModel):
    def __init__(self, **kwargs):
        kwargs['backbone'] = 'resnet50'
        kwargs['channels'] = [768, 256, 128]
        kwargs['get_layer1_feature'] = True
        kwargs['get_layer2_feature'] = True
        kwargs['get_layer3_feature'] = True
        kwargs['get_layer4_feature'] = True
        super(Resnet50EastDetectionModel, self).__init__(**kwargs)

    def forward(self, image):
        return super(Resnet50EastDetectionModel, self).forward(image)


class Resnet101EastDetectionModel(EastDetectionModel):
    def __init__(self, **kwargs):
        kwargs['backbone'] = 'resnet101'
        kwargs['channels'] = [768, 256, 128]
        kwargs['get_layer1_feature'] = True
        kwargs['get_layer2_feature'] = True
        kwargs['get_layer3_feature'] = True
        kwargs['get_layer4_feature'] = True
        super(Resnet101EastDetectionModel, self).__init__(**kwargs)

    def forward(self, image):
        return super(Resnet101EastDetectionModel, self).forward(image)


class Resnet152EastDetectionModel(EastDetectionModel):
    def __init__(self, **kwargs):
        kwargs['backbone'] = 'resnet152'
        kwargs['channels'] = [768, 256, 128]
        kwargs['get_layer1_feature'] = True
        kwargs['get_layer2_feature'] = True
        kwargs['get_layer3_feature'] = True
        kwargs['get_layer4_feature'] = True
        super(Resnet152EastDetectionModel, self).__init__(**kwargs)

    def forward(self, image):
        return super(Resnet152EastDetectionModel, self).forward(image)


class Fishnet99PseDetectionModel(PseDetectionModel):
    def __init__(self, **kwargs):
        kwargs['backbone'] = 'fishnet99'
        kwargs['channels'] = [320, 832, 1600, 1056]
        super(Fishnet99PseDetectionModel, self).__init__(**kwargs)

    def forward(self, image):
        return super(Fishnet99PseDetectionModel, self).forward(image)


class Fishnet150PseDetectionModel(PseDetectionModel):
    def __init__(self, **kwargs):
        kwargs['backbone'] = 'fishnet150'
        kwargs['channels'] = [320, 832, 1600, 1056]
        super(Fishnet150PseDetectionModel, self).__init__(**kwargs)

    def forward(self, image):
        return super(Fishnet150PseDetectionModel, self).forward(image)


class Resnet18PseDetectionMoel(PseDetectionModel):
    def __init__(self, **kwargs):
        kwargs['backbone'] = 'resnet18'
        kwargs['channels'] = [64, 128, 256, 512]
        kwargs['get_layer1_feature'] = True
        kwargs['get_layer2_feature'] = True
        kwargs['get_layer3_feature'] = True
        kwargs['get_layer4_feature'] = True
        super(Resnet18PseDetectionMoel, self).__init__(**kwargs)

    def forward(self, image):
        return super(Resnet18PseDetectionMoel, self).forward(image)


class Resnet50PseDetectionModel(PseDetectionModel):
    def __init__(self, **kwargs):
        kwargs['backbone'] = 'resnet50'
        kwargs['channels'] = [64, 128, 256, 512]
        kwargs['get_layer1_feature'] = True
        kwargs['get_layer2_feature'] = True
        kwargs['get_layer3_feature'] = True
        kwargs['get_layer4_feature'] = True
        super(Resnet50PseDetectionModel, self).__init__(**kwargs)

    def forward(self, image):
        return super(Resnet50PseDetectionModel, self).forward(image)


class Resnet101PseDetectionModel(PseDetectionModel):
    def __init__(self, **kwargs):
        kwargs['backbone'] = 'resnet101'
        kwargs['channels'] = [64, 128, 256, 512]
        kwargs['get_layer1_feature'] = True
        kwargs['get_layer2_feature'] = True
        kwargs['get_layer3_feature'] = True
        kwargs['get_layer4_feature'] = True
        super(Resnet101PseDetectionModel, self).__init__(**kwargs)

    def forward(self, image):
        return super(Resnet101PseDetectionModel, self).forward(image)


class Resnet152PseDetectionModel(PseDetectionModel):
    def __init__(self, **kwargs):
        kwargs['backbone'] = 'resnet152'
        kwargs['channels'] = [64, 128, 256, 512]
        kwargs['get_layer1_feature'] = True
        kwargs['get_layer2_feature'] = True
        kwargs['get_layer3_feature'] = True
        kwargs['get_layer4_feature'] = True
        super(Resnet152PseDetectionModel, self).__init__(**kwargs)

    def forward(self, image):
        return super(Resnet152PseDetectionModel, self).forward(image)


class EastDetectionLoss(nn.Module):
    def __init__(self):
        super(EastDetectionLoss, self).__init__()
        return

    def forward(self, y_true_cls, y_pred_cls, y_true_geo, y_pred_geo, training_mask):
        l_cls = self.__dice_coefficient(y_true_cls, y_pred_cls)
        # scale classification loss to match the iou loss part
        d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = torch.split(y_true_geo, 1, 1)
        d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = torch.split(y_pred_geo, 1, 1)
        area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
        area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
        w_union = torch.min(d2_gt, d2_pred) + torch.min(d4_gt, d4_pred)
        h_union = torch.min(d1_gt, d1_pred) + torch.min(d3_gt, d3_pred)
        area_intersect = w_union * h_union
        area_union = area_gt + area_pred - area_intersect
        l_aabb = -torch.log((area_intersect + 0.01) / (area_union + 0.01))
        l_theta = 1 - torch.cos(theta_pred - theta_gt)
        l_g = l_aabb + 20 * l_theta

        geo_loss = 100 * torch.mean(l_g * y_true_cls * training_mask)
        l_aabb = 100 * torch.mean(l_aabb * y_true_cls * training_mask)
        l_theta = 100 * torch.mean(l_theta * y_true_cls * training_mask)
        l_cls = 2 * l_cls
        loss = geo_loss + l_cls

        return loss, l_aabb, l_theta, l_cls

    @staticmethod
    def __dice_coefficient(y_true_cls, y_pred_cls):
        '''
        dice loss
        :param y_true_cls:
        :param y_pred_cls:
        :param training_mask:
        :return:
        '''
        eps = 1e-5
        intersection = torch.sum(y_true_cls * y_pred_cls)
        union = torch.sum(y_true_cls) + torch.sum(y_pred_cls) + eps
        loss = 1. - (2 * intersection / union)

        return loss


class PseDetectionLoss(nn.Module):
    def __init__(self, **kwargs):
        """Implement PSE Loss.
        """
        super(PseDetectionLoss, self).__init__()
        reduction = get('reduction', kwargs, 'mean')
        assert reduction in ['mean', 'sum'], " reduction must in ['mean','sum']"
        self.Lambda = get('Lambda', kwargs, 0.7)
        self.ratio = get('ratio', kwargs, 3)
        self.reduction = reduction

    def forward(self, outputs, labels, training_masks):
        texts = outputs[:, -1, :, :]
        kernels = outputs[:, :-1, :, :]
        gt_texts = labels[:, -1, :, :]
        gt_kernels = labels[:, :-1, :, :]

        selected_masks = self.ohem_batch(texts, gt_texts, training_masks)
        selected_masks = selected_masks.to(outputs.device)

        loss_text = self.dice_loss(texts, gt_texts, selected_masks)

        loss_kernels = []
        mask0 = torch.sigmoid(texts).data.cpu().numpy()
        mask1 = training_masks.data.cpu().numpy()
        selected_masks = ((mask0 > 0.5) & (mask1 > 0.5)).astype('float32')
        selected_masks = torch.from_numpy(selected_masks).float()
        selected_masks = selected_masks.to(outputs.device)
        kernels_num = gt_kernels.size()[1]
        for i in range(kernels_num):
            kernel_i = kernels[:, i, :, :]
            gt_kernel_i = gt_kernels[:, i, :, :]
            loss_kernel_i = self.dice_loss(kernel_i, gt_kernel_i, selected_masks)
            loss_kernels.append(loss_kernel_i)
        loss_kernels = torch.stack(loss_kernels).mean(0)
        if self.reduction == 'mean':
            loss_text = loss_text.mean()
            loss_kernels = loss_kernels.mean()
        elif self.reduction == 'sum':
            loss_text = loss_text.sum()
            loss_kernels = loss_kernels.sum()

        loss = self.Lambda * loss_text + (1 - self.Lambda) * loss_kernels
        return loss_text, loss_kernels, loss

    def dice_loss(self, input, target, mask):
        input = torch.sigmoid(input)

        input = input.contiguous().view(input.size()[0], -1)
        target = target.contiguous().view(target.size()[0], -1)
        mask = mask.contiguous().view(mask.size()[0], -1)

        input = input * mask
        target = target * mask

        a = torch.sum(input * target, 1)
        b = torch.sum(input * input, 1) + 0.001
        c = torch.sum(target * target, 1) + 0.001
        d = (2 * a) / (b + c)
        return 1 - d

    def ohem_single(self, score, gt_text, training_mask):
        pos_num = (int)(np.sum(gt_text > 0.5)) - (int)(np.sum((gt_text > 0.5) & (training_mask <= 0.5)))

        if pos_num == 0:
            # selected_mask = gt_text.copy() * 0 # may be not good
            selected_mask = training_mask
            selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
            return selected_mask

        neg_num = (int)(np.sum(gt_text <= 0.5))
        neg_num = (int)(min(pos_num * 3, neg_num))

        if neg_num == 0:
            selected_mask = training_mask
            selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
            return selected_mask

        neg_score = score[gt_text <= 0.5]
        neg_score_sorted = np.sort(-neg_score)
        threshold = -neg_score_sorted[neg_num - 1]
        selected_mask = ((score >= threshold) | (gt_text > 0.5)) & (training_mask > 0.5)
        selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
        return selected_mask

    def ohem_batch(self, scores, gt_texts, training_masks):
        scores = scores.data.cpu().numpy()
        gt_texts = gt_texts.data.cpu().numpy()
        training_masks = training_masks.data.cpu().numpy()

        selected_masks = []
        for i in range(scores.shape[0]):
            selected_masks.append(self.ohem_single(scores[i, :, :], gt_texts[i, :, :], training_masks[i, :, :]))

        selected_masks = np.concatenate(selected_masks, 0)
        selected_masks = torch.from_numpy(selected_masks).float()

        return selected_masks

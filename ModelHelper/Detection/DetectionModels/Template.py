from ModelHelper.Common.CommonModels.Template import AbstractTemplate
from ModelHelper.Common.CommonUtils import get, get_valid, generate_log
from ModelHelper.Common.CommonUtils.HandleImage import get_img_list
from ModelHelper.Common.CommonUtils.HandleText import list2txt
from ModelHelper.Common.CommonUtils.Wrapper import time_consume, config
from ModelHelper.Detection.DetectionUtils.DataAugmentation import DetectionCompose
from ModelHelper.Detection.DetectionModels.ModelFactory import DetectionModelFactory
from ModelHelper.Detection.DetectionModels.Dataset import EastDataset, PseDataset
from ModelHelper.Detection.DetectionUtils import DataAugmentation
from ModelHelper.Detection.DetectionModels import EastDetectionLoss, PseDetectionLoss
from ModelHelper.Detection.DetectionUtils.Evaluation import f1score
from ModelHelper.Detection.DetectionModels.east_eval.eval_utils import detect, sort_poly
from ModelHelper.Detection.Component.pse import decode as pse_decode
from torchvision import transforms
import torch
import os
import cv2
import numpy as np


class EastDetectionTemplate(AbstractTemplate):
    def __int__(self, **kwargs):
        super(EastDetectionTemplate, self).__init__(**kwargs)

    def init_model(self, **kwargs):
        return super(EastDetectionTemplate, self).init_model(model_factory=DetectionModelFactory(), **kwargs)

    def init_trainloader(self, **kwargs):
        super(EastDetectionTemplate, self).init_trainloader(**kwargs)
        train_transforms = get('train_transforms', kwargs, None)
        if train_transforms is None:
            train_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        kwargs['transforms'] = train_transforms

        train_detection_transforms = get('train_detection_transforms', kwargs, None)
        if train_detection_transforms is None:
            random_crop_threshold = get('random_crop_threshold', kwargs, (1, 4))
            random_crop_size = get('random_crop_size', kwargs, 768)
            center_rotate_threshold = get('center_rotate_threshold', kwargs, (-30, 30))
            flip_type = get('flip_type', kwargs, 'Horizontal')
            flip_chance = get('flip_chance', kwargs, 0.5)

            train_detection_transforms = DetectionCompose([
                DataAugmentation.Flip(flip_type, flip_chance),
                DataAugmentation.CenterRotate(center_rotate_threshold),
                DataAugmentation.RandomCrop(random_crop_threshold, random_crop_size)
            ])
        kwargs['detection_transforms'] = train_detection_transforms
        train_folder = get_valid('train_folder', kwargs)
        train_dataset = EastDataset(folder=train_folder, **kwargs)
        train_batch = get('train_batch', kwargs, 4)
        train_worker = get('train_worker', kwargs, 8)
        drop_last = get('drop_last', kwargs, True)
        shuffle = get('shuffle', kwargs, True)
        if self.gpu is None:
            batch_size = train_batch
        else:
            batch_size = train_batch * len(self.gpu)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   num_workers=train_worker,
                                                   drop_last=drop_last,
                                                   shuffle=shuffle)
        train_data_num = len(train_dataset)
        print('Generate train data loader, train data folder: {}, train data num: {}'.format(train_folder,
                                                                                             train_data_num))
        return train_loader

    def init_testloader(self, **kwargs):
        super(EastDetectionTemplate, self).init_testloader(**kwargs)
        test_transforms = get('test_transforms', kwargs, None)
        if test_transforms is None:
            test_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        kwargs['transforms'] = test_transforms
        test_folder = get_valid('test_folder', kwargs)
        assert os.path.exists(test_folder)
        test_dataset = EastDataset(folder=test_folder,
                                   **kwargs)
        test_data_num = len(test_dataset)
        test_batch = get('test_batch', kwargs, 4)
        test_worker = get('test_worker', kwargs, 8)
        if self.gpu is None:
            batch_size = test_batch
        else:
            batch_size = test_batch * len(self.gpu)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size,
                                                  num_workers=test_worker)
        print('Generate test data loader, test data folder: {}, train data num: {}'.format(test_folder,
                                                                                           test_data_num))
        return test_loader

    def init_optimizer(self, **kwargs):
        model = get_valid('model', kwargs)
        lr = get('lr', kwargs, 0.001)
        weight_decay = get('weight_decay', kwargs, 1e-4)
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def init_criterion(self, **kwargs):
        super(EastDetectionTemplate, self).init_criterion(**kwargs)
        return EastDetectionLoss()

    @time_consume
    def train_model(self, **kwargs):
        super(EastDetectionTemplate, self).train_model(**kwargs)
        model = get_valid('model', kwargs)
        model = model.train()
        optimizer = get_valid('optimizer', kwargs)
        epoch = get_valid('epoch', kwargs)
        criterion = get_valid('criterion', kwargs)
        train_loader = get_valid('train_loader', kwargs)

        total_loss = 0
        aabb_loss = 0
        theta_loss = 0
        cls_loss = 0
        iter_num = len(train_loader)
        if iter_num == 0:
            raise RuntimeError('training data num < batch num!')
        for data in train_loader:
            img = data['img']
            score_map = data['score_map']
            geo_map = data['geo_map']
            training_mask = data['training_mask']
            if self.gpu is not None:
                img = img.cuda()
                score_map = score_map.cuda()
                geo_map = geo_map.cuda()
                training_mask = training_mask.cuda()

            f_score, f_geometry = model(image=img)
            geo_map = geo_map.permute(0, 3, 1, 2).contiguous()
            loss, l_aabb, l_theta, l_cls = criterion(score_map, f_score, geo_map, f_geometry, training_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.data.item()
            aabb_loss += l_aabb.data.item()
            theta_loss += l_theta.data.item()
            cls_loss += l_cls.data.item()

        avg_loss = total_loss / iter_num
        avg_aabb_loss = aabb_loss / iter_num
        avg_theta_loss = theta_loss / iter_num
        avg_cls_loss = cls_loss / iter_num

        log = generate_log(epoch=epoch,
                           name='Train',
                           avg_loss=avg_loss,
                           avg_aabb_loss=avg_aabb_loss,
                           avg_theta_loss=avg_theta_loss,
                           avg_cls_loss=avg_cls_loss)
        return log, avg_loss

    @time_consume
    def test_model(self, **kwargs):
        super(EastDetectionTemplate, self).test_model(**kwargs)
        model = get_valid('model', kwargs)
        model = model.eval()
        epoch = get_valid('epoch', kwargs)
        criterion = get_valid('criterion', kwargs)
        test_loader = get_valid('test_loader', kwargs)

        total_loss = 0
        aabb_loss = 0
        theta_loss = 0
        cls_loss = 0
        iter_num = len(test_loader)
        with torch.no_grad():
            for data in test_loader:
                img = data['img']
                score_map = data['score_map']
                geo_map = data['geo_map']
                training_mask = data['training_mask']
                # img_path = data['img_path']
                if self.gpu is not None:
                    img = img.cuda()
                    score_map = score_map.cuda()
                    geo_map = geo_map.cuda()
                    training_mask = training_mask.cuda()

                f_score, f_geometry = model(image=img)
                geo_map = geo_map.permute(0, 3, 1, 2).contiguous()
                loss, l_aabb, l_theta, l_cls = criterion(score_map, f_score, geo_map, f_geometry, training_mask)

                total_loss += loss.data.item()
                aabb_loss += l_aabb.data.item()
                theta_loss += l_theta.data.item()
                cls_loss += l_cls.data.item()

        avg_loss = total_loss / iter_num
        avg_aabb_loss = aabb_loss / iter_num
        avg_theta_loss = theta_loss / iter_num
        avg_cls_loss = cls_loss / iter_num

        log = generate_log(epoch=epoch,
                           name='Test',
                           avg_loss=avg_loss,
                           avg_aabb_loss=avg_aabb_loss,
                           avg_theta_loss=avg_theta_loss,
                           avg_cls_loss=avg_cls_loss)

        return log, avg_loss

    def eval_model(self, **kwargs):
        score_map_thresh = get('score_map_thresh', kwargs, 0.7)
        box_thresh = get('box_thresh', kwargs, 0.1)
        nms_thresh = get('nms_thresh', kwargs, 0.2)
        model = get_valid('model', kwargs)
        model = model.eval()

        eval_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        test_folder = get_valid('test_folder', kwargs)
        pred_folder = get_valid('pred_folder', kwargs)
        img_list = get_img_list(test_folder)
        for img in img_list:
            img_path = os.path.join(test_folder, img)
            image = cv2.imread(img_path)
            image = image.astype(np.float32)
            image = eval_transforms(image)
            image = image.unsqueeze(0)
            if self.gpu is not None:
                image = image.cuda()

            with torch.no_grad():
                score, geometry = model(image=image)

            score = score.permute(0, 2, 3, 1)
            geometry = geometry.permute(0, 2, 3, 1)
            score = score.data.cpu().numpy()
            geometry = geometry.data.cpu().numpy()
            boxes = detect(score_map=score,
                           geo_map=geometry,
                           score_map_thresh=score_map_thresh,
                           box_thresh=box_thresh,
                           nms_thresh=nms_thresh)

            txt_name = img.split('.')[0] + '.txt'
            txt_path = os.path.join(pred_folder, txt_name)

            pred_list = list()
            if boxes is not None:
                boxes = boxes[:, :8].reshape(-1, 4, 2)
                for box in boxes:
                    box = sort_poly(box.astype(np.int32))
                    pred = '{},{},{},{},{},{},{},{}'.format(box[0, 0], box[0, 1], box[1, 0], box[1, 1],
                                                            box[2, 0], box[2, 1], box[3, 0], box[3, 1])
                    pred_list.append(pred)
            list2txt(pred_list, txt_path)
        iou_threshold = get('iou_threshold', kwargs, 0.5)
        score, total_label_num, total_pred_num, total_correct_num = f1score(test_folder, pred_folder, iou_threshold)
        if total_label_num == 0 or total_correct_num == 0 or total_pred_num == 0:
            precision = 0
            recall = 0
        else:
            precision = total_correct_num / total_label_num
            recall = total_correct_num / total_pred_num
        print('f1score: {}, precision={}/{}={}, recall={}/{}={}, iou threshold: {};'.format(score, total_correct_num,
                                                                                            total_label_num, precision,
                                                                                            total_correct_num,
                                                                                            total_pred_num, recall,
                                                                                            iou_threshold))
        return score

    def load_model(self, **kwargs):
        return super(EastDetectionTemplate, self).load_model(**kwargs)

    def save_model(self, **kwargs):
        return super(EastDetectionTemplate, self).save_model(**kwargs)

    @config
    def run(self, **kwargs):
        super(EastDetectionTemplate, self).run(**kwargs)


class PseDetectionTemplate(AbstractTemplate):
    def __init__(self, **kwargs):
        super(PseDetectionTemplate, self).__init__(**kwargs)

    def init_model(self, **kwargs):
        return super(PseDetectionTemplate, self).init_model(model_factory=DetectionModelFactory(), **kwargs)

    def init_trainloader(self, **kwargs):
        super(PseDetectionTemplate, self).init_trainloader(**kwargs)
        train_transforms = get('train_transforms', kwargs, None)
        if train_transforms is None:
            train_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        kwargs['transforms'] = train_transforms

        train_detection_transforms = get('train_detection_transforms', kwargs, None)
        if train_detection_transforms is None:
            random_crop_threshold = get('random_crop_threshold', kwargs, (1, 4))
            random_crop_size = get('random_crop_size', kwargs, 768)
            center_rotate_threshold = get('center_rotate_threshold', kwargs, (-30, 30))
            flip_type = get('flip_type', kwargs, 'Horizontal')
            flip_chance = get('flip_chance', kwargs, 0.5)

            train_detection_transforms = DetectionCompose([
                DataAugmentation.Flip(flip_type, flip_chance),
                DataAugmentation.CenterRotate(center_rotate_threshold),
                DataAugmentation.RandomCrop(random_crop_threshold, random_crop_size)
            ])
        kwargs['detection_transforms'] = train_detection_transforms
        train_folder = get_valid('train_folder', kwargs)
        train_dataset = PseDataset(folder=train_folder, **kwargs)
        train_batch = get('train_batch', kwargs, 4)
        train_worker = get('train_worker', kwargs, 8)
        drop_last = get('drop_last', kwargs, True)
        shuffle = get('shuffle', kwargs, True)
        if self.gpu is None:
            batch_size = train_batch
        else:
            batch_size = train_batch * len(self.gpu)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   num_workers=train_worker,
                                                   drop_last=drop_last,
                                                   shuffle=shuffle)
        train_data_num = len(train_dataset)
        print('Generate train data loader, train data folder: {}, train data num: {}'.format(train_folder,
                                                                                             train_data_num))
        return train_loader

    def init_testloader(self, **kwargs):
        super(PseDetectionTemplate, self).init_testloader(**kwargs)
        test_transforms = get('test_transforms', kwargs, None)
        if test_transforms is None:
            test_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        kwargs['transforms'] = test_transforms
        test_folder = get_valid('test_folder', kwargs)
        assert os.path.exists(test_folder)
        test_dataset = PseDataset(folder=test_folder, **kwargs)
        test_data_num = len(test_dataset)
        test_batch = get('test_batch', kwargs, 4)
        test_worker = get('test_worker', kwargs, 8)
        if self.gpu is None:
            batch_size = test_batch
        else:
            batch_size = test_batch * len(self.gpu)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size,
                                                  num_workers=test_worker)
        print('Generate test data loader, test data folder: {}, train data num: {}'.format(test_folder,
                                                                                           test_data_num))
        return test_loader

    def init_optimizer(self, **kwargs):
        super(PseDetectionTemplate, self).init_optimizer(**kwargs)
        model = get_valid('model', kwargs)
        lr = get('lr', kwargs, 0.001)
        weight_decay = get('weight_decay', kwargs, 1e-4)
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def init_criterion(self, **kwargs):
        super(PseDetectionTemplate, self).init_criterion(**kwargs)
        return PseDetectionLoss(**kwargs)

    @time_consume
    def train_model(self, **kwargs):
        super(PseDetectionTemplate, self).train_model(**kwargs)
        model = get_valid('model', kwargs)
        model.train()
        optimizer = get_valid('optimizer', kwargs)
        epoch = get_valid('epoch', kwargs)
        criterion = get_valid('criterion', kwargs)
        train_loader = get_valid('train_loader', kwargs)

        total_loss = 0
        total_text_loss = 0
        total_kernels_loss = 0
        iter_num = len(train_loader)
        if iter_num == 0:
            raise RuntimeError('training data num < batch num!')
        for data in train_loader:
            img = data['img']
            score_maps = data['score_maps']
            training_mask = data['training_mask']
            if self.gpu is not None:
                img = img.cuda()
                score_maps = score_maps.cuda()
                training_mask = training_mask.cuda()

            output = model(image=img)
            loss_text, loss_kernels, loss = criterion(output, score_maps, training_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.data.item()
            total_text_loss += loss_text.data.item()
            total_kernels_loss += loss_kernels.data.item()

        avg_loss = total_loss / iter_num
        avg_text_loss = total_text_loss / iter_num
        avg_kernels_loss = total_kernels_loss / iter_num

        log = generate_log(epoch=epoch,
                           name='Train',
                           avg_loss=avg_loss,
                           avg_text_loss=avg_text_loss,
                           avg_kernels_loss=avg_kernels_loss)
        return log, avg_loss

    @time_consume
    def test_model(self, **kwargs):
        super(PseDetectionTemplate, self).test_model(**kwargs)
        model = get_valid('model', kwargs)
        model.train()
        epoch = get_valid('epoch', kwargs)
        criterion = get_valid('criterion', kwargs)
        test_loader = get_valid('test_loader', kwargs)

        total_loss = 0
        total_text_loss = 0
        total_kernels_loss = 0
        iter_num = len(test_loader)
        with torch.no_grad():
            for data in test_loader:
                img = data['img']
                score_maps = data['score_maps']
                training_mask = data['training_mask']
                if self.gpu is not None:
                    img = img.cuda()
                    score_maps = score_maps.cuda()
                    training_mask = training_mask.cuda()

                output = model(image=img)
                loss_text, loss_kernels, loss = criterion(output, score_maps, training_mask)
                total_loss += loss.data.item()
                total_text_loss += loss_text.data.item()
                total_kernels_loss += loss_kernels.data.item()

        avg_loss = total_loss / iter_num
        avg_text_loss = total_text_loss / iter_num
        avg_kernels_loss = total_kernels_loss / iter_num

        log = generate_log(epoch=epoch,
                           name='Test',
                           avg_loss=avg_loss,
                           avg_text_loss=avg_text_loss,
                           avg_kernels_loss=avg_kernels_loss)
        return log, avg_loss

    def eval_model(self, **kwargs):
        model = get_valid('model', kwargs)
        model.eval()

        eval_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        test_folder = get_valid('test_folder', kwargs)
        pred_folder = get_valid('pred_folder', kwargs)
        scale = get('scale', kwargs, 4)
        img_list = get_img_list(test_folder)
        for img in img_list:
            img_path = os.path.join(test_folder, img)
            image = cv2.imread(img_path)
            image = image.astype(np.float32)
            image = eval_transforms(image)
            image = image.unsqueeze(0)
            if self.gpu is not None:
                image = image.cuda()

            with torch.no_grad():
                output = model(image=image)
            preds, boxes_list = pse_decode(output[0], scale)

            pred_list = list()
            if len(boxes_list)>0:
                for box in boxes_list:
                    box = sort_poly(box.astype(np.int32))
                    pred = '{},{},{},{},{},{},{},{}'.format(box[0, 0], box[0, 1], box[1, 0], box[1, 1],
                                                            box[2, 0], box[2, 1], box[3, 0], box[3, 1])
                    pred_list.append(pred)
            txt_name = img.split('.')[0] + '.txt'
            txt_path = os.path.join(pred_folder, txt_name)
            list2txt(boxes_list, txt_path)


        iou_threshold = get('iou_threshold', kwargs, 0.5)
        score, total_label_num, total_pred_num, total_correct_num = f1score(test_folder, pred_folder, iou_threshold)
        if total_label_num == 0 or total_correct_num == 0 or total_pred_num == 0:
            precision = 0
            recall = 0
        else:
            precision = total_correct_num / total_label_num
            recall = total_correct_num / total_pred_num
        print('f1score: {}, precision={}/{}={}, recall={}/{}={}, iou threshold: {};'.format(score, total_correct_num,
                                                                                            total_label_num, precision,
                                                                                            total_correct_num,
                                                                                            total_pred_num, recall,
                                                                                            iou_threshold))
        return score

    def load_model(self, **kwargs):
        return super(PseDetectionTemplate, self).load_model(**kwargs)

    def save_model(self, **kwargs):
        return super(PseDetectionTemplate, self).save_model(**kwargs)

    @config
    def run(self, **kwargs):
        super(PseDetectionTemplate, self).run(**kwargs)


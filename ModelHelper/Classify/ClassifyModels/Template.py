from ModelHelper.Common.CommonModels.Template import AbstractTemplate
from ModelHelper.Common.CommonUtils import get, get_valid, generate_log
from ModelHelper.Common.CommonUtils.ImageAugmentation import Padding, RandomRotate, SaltNoise, RandomDistort
from ModelHelper.Common.CommonUtils.Wrapper import time_consume, config
from ModelHelper.Classify.ClassifyModels.AbstractModels import NTSClassifyModel
from ModelHelper.Classify.ClassifyModels.ModelFactory import ClassifyModelFactory
from ModelHelper.Classify.ClassifyModels.Datasets import ClassifyDataset
from torchvision import transforms
import os
import torch
from torch.optim.lr_scheduler import MultiStepLR
from collections import OrderedDict
import torch.nn as nn


class ClassifyTemplate(AbstractTemplate):
    def __init__(self, **kwargs):
        super(ClassifyTemplate, self).__init__(**kwargs)

    def init_model(self, **kwargs):
        return super(ClassifyTemplate, self).init_model(model_factory=ClassifyModelFactory(), **kwargs)

    def init_trainloader(self, **kwargs):
        super(ClassifyTemplate, self).init_trainloader(**kwargs)
        train_transforms = get('train_transforms', kwargs)
        if train_transforms is None:
            input_size = get('input_size', kwargs, 600)
            crop_size = get('crop_size', kwargs, 448)

            train_transforms = transforms.Compose([
                RandomDistort(),
                RandomRotate(),
                Padding(input_size),
                SaltNoise(0.05),
                transforms.RandomCrop((crop_size, crop_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        train_folder = get_valid('train_folder', kwargs)
        assert os.path.exists(train_folder)
        train_folder = train_folder
        train_dataset = ClassifyDataset(folder=train_folder,
                                        is_training=True,
                                        transforms=train_transforms,
                                        **kwargs)
        train_batch = get('train_batch', kwargs, 16)
        train_worker = get('train_worker', kwargs, 8)
        drop_last = get('drop_last', kwargs, True)
        shuffle = get('shuffle', kwargs, True)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=train_batch,
                                                   num_workers=train_worker,
                                                   drop_last=drop_last,
                                                   shuffle=shuffle)
        train_data_num = len(train_dataset)
        self.train_data_num = train_data_num
        print('Generate train data loader, train data folder: {}, train data num: {}'.format(train_folder,
                                                                                             train_data_num))
        return train_loader

    def init_testloader(self, **kwargs):
        super(ClassifyTemplate, self).init_testloader(**kwargs)
        test_transforms = get('test_transforms', kwargs)
        if test_transforms is None:
            input_size = get('input_size', kwargs, 600)
            crop_size = get('crop_size', kwargs, 448)

            test_transforms = transforms.Compose([
                Padding(input_size),
                transforms.CenterCrop((crop_size, crop_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        test_folder = get('test_folder', kwargs)
        assert os.path.exists(test_folder)
        test_folder = test_folder
        test_datset = ClassifyDataset(folder=test_folder,
                                      is_training=False,
                                      transforms=test_transforms,
                                      **kwargs)

        test_batch = get('test_batch', kwargs, 16)
        test_worker = get('test_worker', kwargs, 8)

        test_loader = torch.utils.data.DataLoader(test_datset,
                                                  batch_size=test_batch,
                                                  num_workers=test_worker)
        test_data_num = len(test_datset)
        self.test_data_num = test_data_num
        print('generate test data loader, test data folder: {}, test data num:{}'.format(test_folder,
                                                                                         test_data_num))
        return test_loader

    def init_optimizer(self, **kwargs):
        super(ClassifyTemplate, self).init_optimizer(**kwargs)
        model = get_valid('model', kwargs)
        lr = get('lr', kwargs, 0.001)
        weight_decay = get('weight_decay', kwargs, 1e-4)
        raw_optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        return raw_optimizer

    def init_criterion(self, **kwargs):
        super(ClassifyTemplate, self).init_criterion(**kwargs)
        return torch.nn.CrossEntropyLoss()

    @time_consume
    def train_model(self, **kwargs):
        super(ClassifyTemplate, self).train_model(**kwargs)
        optimizer = get_valid('optimizer', kwargs)
        train_loader = get_valid('train_loader', kwargs)
        criterion = get_valid('criterion', kwargs)
        model = get_valid('model', kwargs)
        epoch = get_valid('epoch', kwargs)
        sum_loss = 0
        iter_num = len(train_loader)
        log_step = iter_num // 100
        log_step = max(1, log_step)

        model = model.train()

        for idx, data in enumerate(train_loader):
            img, label = data[0], data[1]
            if self.gpu is not None:
                img = img.cuda()
                label = label.cuda()
                model = model.cuda()

            optimizer.zero_grad()
            pred = model(image=img)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()

            if idx % log_step == 0 and idx != 0:
                finish_percent = int(idx / iter_num * 100)
                print('Train: finish {}%, loss: {}'.format(finish_percent, loss.data.item()))
            sum_loss += loss.data.item()
        avg_loss = sum_loss / iter_num
        log = generate_log(epoch=epoch,
                           name='Train',
                           avg_loss=avg_loss)
        return log, avg_loss

    @time_consume
    def test_model(self, **kwargs):
        super(ClassifyTemplate, self).test_model(**kwargs)
        test_loader = get_valid('test_loader', kwargs)
        criterion = get_valid('criterion', kwargs)
        model = get_valid('model', kwargs)
        epoch = get_valid('epoch', kwargs)
        sum_loss = 0
        correct = 0
        iter_num = len(test_loader)

        log_step = iter_num // 100
        log_step = max(1, log_step)

        model = model.eval()

        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                img, label = data[0], data[1]
                if self.gpu is not None:
                    img = img.cuda()
                    label = label.cuda()
                    model = model.cuda()
                pred = model(image=img)
                loss = criterion(pred, label)
                _, pred_cls = torch.max(pred, 1)
                correct += torch.sum(pred_cls.data == label.data)
                sum_loss += loss.data.item()

                if idx % log_step == 0 and idx != 0:
                    finish_percent = int(idx / iter_num * 100)
                    print('Test: finish {}%, loss: {}'.format(finish_percent, loss.data.item()))

        avg_loss = sum_loss / iter_num
        acc = int(correct) / self.test_data_num
        log = generate_log(epoch=epoch,
                           name='Test',
                           avg_loss=avg_loss,
                           acc=acc)
        return log, avg_loss, acc

    def eval_model(self, **kwargs):
        super(ClassifyTemplate, self).eval_model(**kwargs)
        pass

    def load_model(self, **kwargs):
        model = get_valid('model', kwargs)
        checkpoint = get_valid('checkpoint', kwargs)
        assert os.path.exists(checkpoint)

        if self.use_gpu is False:
            ckpt = torch.load(checkpoint, map_location='cpu')
        else:
            ckpt = torch.load(checkpoint)

        new_state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        start_epoch = ckpt['epoch'] + 1
        loss = ckpt['loss']
        acc = ckpt['acc']
        return model, start_epoch, acc, loss

    def save_model(self, **kwargs):
        model = get_valid('model', kwargs)
        ckpt_folder = get_valid('ckpt_folder', kwargs)
        epoch = get_valid('epoch', kwargs)
        loss = get_valid('loss', kwargs)
        eval_type = get('eval_type', kwargs, 'acc')
        acc = get('acc', kwargs)
        acc = round(acc, 4)
        model_name = get('model_name', kwargs, '')

        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'eval_type': eval_type,
            'acc': acc,
            'loss': loss
        }

        loss = round(loss, 4)

        ckpt_name = '{}epoch{}_{}{}_loss{}.pth'.format(model_name, epoch, eval_type, acc, loss)
        ckpt_path = os.path.join(ckpt_folder, ckpt_name)
        torch.save(state, ckpt_path)
        print('Save checkpoint: {}'.format(ckpt_path))
        return ckpt_path

    @config
    def run(self, **kwargs):
        model = self.init_model(**kwargs)

        train_loader = self.init_trainloader(model=model, **kwargs)
        test_loader = self.init_testloader(model=model, **kwargs)

        optimizer = self.init_optimizer(model=model, **kwargs)
        criterion = self.init_criterion(**kwargs)

        output_folder = get_valid('output_folder', kwargs)
        ckpt_folder = os.path.join(output_folder, 'ckpt')
        if not os.path.exists(ckpt_folder):
            os.makedirs(ckpt_folder)
        log_path = os.path.join(output_folder, 'log.txt')
        checkpoint = get('checkpoint', kwargs)
        if checkpoint is not None:
            model, start_epoch, max_acc, min_loss = self.load_model(model=model, **kwargs)
        else:
            start_epoch = 0
            max_acc = 0
            min_loss = 999999

        if self.gpu is not None:
            model.cuda()
            model = nn.DataParallel(model, device_ids=self.gpu)

        epoch_num = get('epoch_num', kwargs, 1000)
        do_test = get('do_test', kwargs, True)
        test_step = get('test_step', kwargs, 1)
        save_by_loss = get('save_by_loss', kwargs, False)

        for epoch in range(start_epoch, epoch_num):
            train_log, train_loss = self.train_model(train_loader=train_loader,
                                                     optimizer=optimizer,
                                                     criterion=criterion,
                                                     model=model,
                                                     epoch=epoch,
                                                     **kwargs)
            log = open(log_path, 'a', encoding=self.encoding)
            log.write(train_log)
            log.close()
            print(train_log)

            if not do_test:
                if epoch % test_step == 0:
                    self.save_model(model=model,
                                    ckpt_folder=ckpt_folder,
                                    epoch=epoch,
                                    loss=train_loss,
                                    acc=0,
                                    **kwargs
                                    )
            else:
                if epoch % test_step == 0:
                    test_log, test_loss, test_acc = self.test_model(test_loader=test_loader,
                                                                    criterion=criterion,
                                                                    model=model,
                                                                    epoch=epoch,
                                                                    **kwargs)
                    if test_acc >= max_acc:
                        max_acc = test_acc
                        self.save_model(model=model,
                                        ckpt_folder=ckpt_folder,
                                        epoch=epoch,
                                        loss=test_loss,
                                        acc=max_acc,
                                        **kwargs)
                    log = open(log_path, 'a', encoding=self.encoding)
                    log.write(test_log)
                    log.close()
                    print(test_log)


class NTSClassifyTemplate(ClassifyTemplate):
    def __init__(self, **kwargs):
        super(NTSClassifyTemplate, self).__init__(**kwargs)

    def init_model(self, **kwargs):
        return super(NTSClassifyTemplate, self).init_model(**kwargs)

    def init_trainloader(self, **kwargs):
        return super(NTSClassifyTemplate, self).init_trainloader(**kwargs)

    def init_testloader(self, **kwargs):
        return super(NTSClassifyTemplate, self).init_testloader(**kwargs)

    def init_optimizer(self, model, **kwargs):
        learning_rate = get('learning_rate', kwargs, 0.001)
        weight_decay = get('weight_decay', kwargs, 1e-4)

        raw_parameters = list(model.backbone.parameters())
        part_parameters = list(model.proposal_net.parameters())
        concat_parameters = list(model.concat_net.parameters())
        partcls_parameters = list(model.partcls_net.parameters())

        raw_optimizer = torch.optim.SGD(raw_parameters, lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
        concat_optimizer = torch.optim.SGD(concat_parameters, lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
        part_optimizer = torch.optim.SGD(part_parameters, lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
        partcls_optimizer = torch.optim.SGD(partcls_parameters, lr=learning_rate, momentum=0.9,
                                            weight_decay=weight_decay)

        schedulers = [MultiStepLR(raw_optimizer, milestones=[60, 100], gamma=0.1),
                      MultiStepLR(concat_optimizer, milestones=[60, 100], gamma=0.1),
                      MultiStepLR(part_optimizer, milestones=[60, 100], gamma=0.1),
                      MultiStepLR(partcls_optimizer, milestones=[60, 100], gamma=0.1)]

        return raw_optimizer, concat_optimizer, part_optimizer, partcls_optimizer, schedulers

    def init_criterion(self, **kwargs):
        return super(NTSClassifyTemplate, self).init_criterion(**kwargs)

    def train_model(self, **kwargs):
        (raw_optimizer, concat_optimizer, part_optimizer, partcls_optimizer, schedulers) = get_valid('optimizer',
                                                                                                     kwargs)
        train_loader = get_valid('train_loader', kwargs)
        criterion = get_valid('criterion', kwargs)
        proposal_num = get('proposal_num', kwargs, 6)
        model = get_valid('model', kwargs)
        epoch = get_valid('epoch', kwargs)
        sum_total_loss = 0
        sum_concat_loss = 0
        sum_raw_loss = 0
        sum_rank_loss = 0
        sum_partcls_loss = 0
        iter_num = len(train_loader)

        log_step = iter_num // 100
        log_step = max(1, log_step)

        for scheduler in schedulers:
            scheduler.step()

        model = model.train()

        for idx, data in enumerate(train_loader):
            img, label = data[0], data[1]
            if self.gpu is not None:
                img = img.cuda()
                label = label.cuda()

            raw_optimizer.zero_grad()
            concat_optimizer.zero_grad()
            part_optimizer.zero_grad()
            partcls_optimizer.zero_grad()

            batch_size = img.size(0)

            raw_logits, concat_logits, part_logits, _, top_n_prob = model(image=img)
            part_loss = NTSClassifyModel.list_loss(part_logits.view(batch_size * proposal_num, -1),
                                                   label.unsqueeze(1).repeat(1, proposal_num).view(-1)).view(
                batch_size, proposal_num)
            raw_loss = criterion(raw_logits, label)
            concat_loss = criterion(concat_logits, label)
            rank_loss = NTSClassifyModel.ranking_loss(top_n_prob, part_loss, proposal_num, **kwargs)
            partcls_loss = criterion(part_logits.view(batch_size * proposal_num, -1),
                                     label.unsqueeze(1).repeat(1, proposal_num).view(-1))

            total_loss = raw_loss + rank_loss + concat_loss + partcls_loss
            total_loss.backward()
            raw_optimizer.step()
            part_optimizer.step()
            concat_optimizer.step()
            partcls_optimizer.step()

            sum_total_loss += total_loss.data.item()
            sum_raw_loss += raw_loss.data.item()
            sum_concat_loss += concat_loss.data.item()
            sum_rank_loss += rank_loss.data.item()
            sum_partcls_loss += partcls_loss.data.item()

            if idx % log_step == 0 and idx != 0:
                finish_percent = int(idx / iter_num * 100)
                print('Train: finish {}%,total loss: {}, raw loss: {}, rank loss: {}, concat loss: {},'
                      ' partcls loss: {}'.format(finish_percent,
                                                 float(total_loss),
                                                 float(raw_loss),
                                                 float(rank_loss),
                                                 float(concat_loss),
                                                 float(partcls_loss)))

        avg_total_loss = sum_total_loss / iter_num
        avg_raw_loss = sum_raw_loss / iter_num
        avg_concat_loss = sum_concat_loss / iter_num
        avg_rank_loss = sum_rank_loss / iter_num
        avg_partcls_loss = sum_partcls_loss / iter_num
        log = generate_log(epoch=epoch,
                           name='Train',
                           avg_total_loss=avg_total_loss,
                           avg_raw_loss=avg_raw_loss,
                           avg_concat_loss=avg_concat_loss,
                           avg_rank_loss=avg_rank_loss,
                           avg_partcls_loss=avg_partcls_loss)
        return log, avg_total_loss

    def test_model(self, **kwargs):
        test_loader = get_valid('test_loader', kwargs)
        criterion = get_valid('criterion', kwargs)
        model = get_valid('model', kwargs)
        epoch = get_valid('epoch', kwargs)
        loss = 0
        correct = 0

        iter_num = len(test_loader)

        log_step = iter_num // 100
        log_step = max(1, log_step)
        model = model.eval()

        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                img, label = data[0], data[1]
                if self.gpu is not None:
                    img = img.cuda()
                    label = label.cuda()
                batch_size = img.size(0)

                _, concat_logits, _, _, _ = model(image=img)
                concat_loss = criterion(concat_logits, label)
                _, concat_predict = torch.max(concat_logits, 1)
                correct += torch.sum(concat_predict.data == label.data)
                loss += concat_loss.item() * batch_size

                if idx % log_step == 0 and idx != 0:
                    finish_percent = int(idx / iter_num * 100)
                    print('Test: finish {}%,concat loss: {},'.format(finish_percent, float(concat_loss.item())))

        loss = loss / self.test_data_num
        acc = float(correct) / self.test_data_num
        log = generate_log(epoch=epoch,
                           name='Test',
                           loss=loss,
                           acc=acc)
        return log, loss, acc

    def eval_model(self, **kwargs):
        super(NTSClassifyTemplate, self).eval_model(**kwargs)

    def load_model(self, **kwargs):
        return super(NTSClassifyTemplate, self).load_model(**kwargs)

    def save_model(self, **kwargs):
        super(NTSClassifyTemplate, self).save_model(**kwargs)

    def run(self, **kwargs):
        super(NTSClassifyTemplate, self).run(use_gpu=self.use_gpu, **kwargs)

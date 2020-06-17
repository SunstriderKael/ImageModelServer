from ModelHelper.Common.CommonModels.Template import AbstractTemplate
from ModelHelper.Common.CommonUtils import get, get_valid, generate_log
from ModelHelper.Common.CommonUtils.Wrapper import config, time_consume
from ModelHelper.Recognition.RecognitionModels.ModelFactory import RecognitionModelFactory
from ModelHelper.Recognition.RecognitionModels.Dataset import SarDataset, MultiDecoderClassifyDataset
from ModelHelper.Common.CommonUtils.ImageAugmentation import RandomDistort, Denoise, Compose
import torch.optim as optim
from torchvision import transforms
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
import random


class SarRecognitionTemplate(AbstractTemplate):
    def __init__(self, **kwargs):
        super(SarRecognitionTemplate, self).__init__(**kwargs)
        word_index_path = get_valid('word_index_path', kwargs)
        self.word2num, self.num2word = self.generate_dict(word_index_path)
        assert len(self.word2num) == len(self.num2word)
        self.class_num = len(self.word2num)
        self.max_len = get('max_len', kwargs, 64)

    @staticmethod
    def generate_dict(word_index_path, encoding='utf-8'):
        word_index = open(word_index_path, 'r', encoding=encoding)
        word2num = dict()
        num2word = dict()
        word2num['SOS'] = 0
        num2word[0] = 'SOS'
        num = 1
        for line in word_index.readlines():
            line = line.replace(' ', '')
            line = line.replace('\n', '')
            for idx in range(len(line)):
                word = line[idx]
                if word not in word2num.keys():
                    word2num[word] = num
                    num2word[num] = word
                    num += 1
        word_index.close()
        return word2num, num2word

    def encode(self, line_list, ignore_index):
        mx = 0
        num = len(line_list)
        new_line_list = list()
        for line in line_list:
            l = len(line)
            if l > self.max_len - 2:
                line = line[0:self.max_len - 2]
                mx = self.max_len - 2
            if l > mx:
                mx = l
            new_line_list.append(line)

        label = np.zeros((num, mx + 2), dtype=np.int32)
        if ignore_index:
            label -= 1

        row = 0
        for line in new_line_list:
            line = line.strip()
            line = line.replace(' ', '')
            col = 1
            for word in line:
                l = self.word2num[word]
                label[row, col] = l
                col += 1
            label[row, 0] = 0
            label[row, col] = 0
            row += 1
        return label

    def decode(self, preds_tensor):
        label_list = list()
        for pred in preds_tensor:
            pred = pred.cpu().numpy()
            label = ''
            for p in pred:
                p = int(p.argmax())
                if p != 0:
                    label += self.num2word[p]
                else:
                    break
            label_list.append(label)
        return label_list

    def init_model(self, **kwargs):
        kwargs['class_num'] = self.class_num
        return super(SarRecognitionTemplate, self).init_model(model_factory=RecognitionModelFactory(), **kwargs)

    def init_trainloader(self, **kwargs):
        super(SarRecognitionTemplate, self).init_trainloader(**kwargs)
        train_transforms = get('train_transforms', kwargs, None)
        if train_transforms is None:
            distort_num = get('distort_num', kwargs, 10)
            distort_ratio = get('distort_ratio', kwargs, 0.1)
            train_transforms = transforms.Compose([
                RandomDistort(distort_num, distort_ratio),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        kwargs['transforms'] = train_transforms
        train_index = get_valid('train_index', kwargs)
        train_folder = get_valid('train_folder', kwargs)
        train_dataset = SarDataset(index_path=train_index, folder=train_folder, **kwargs)
        train_batch = get('train_batch', kwargs, 64)
        train_worker = get('train_worker', kwargs, 8)
        drop_last = get('drop_last', kwargs, True)
        shuffle = get('shuffle', kwargs, True)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=self.get_batch_size(train_batch),
                                                   num_workers=train_worker,
                                                   drop_last=drop_last,
                                                   shuffle=shuffle)
        train_data_num = len(train_dataset)
        print('Analyse train index: {}, train data num: {}!'.format(train_index, train_data_num))
        return train_loader

    def init_testloader(self, **kwargs):
        super(SarRecognitionTemplate, self).init_testloader(**kwargs)
        test_transforms = get('test_transforms', kwargs, None)
        if test_transforms is None:
            test_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        kwargs['transforms'] = test_transforms
        test_index = get_valid('test_index', kwargs)
        test_folder = get_valid('test_folder', kwargs)
        test_dataset = SarDataset(index_path=test_index, folder=test_folder, **kwargs)

        test_batch = get('test_batch', kwargs, 64)
        test_worker = get('test_worker', kwargs, 8)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=self.get_batch_size(test_batch),
                                                  num_workers=test_worker)
        test_data_num = len(test_dataset)
        print('Analyse test index: {}, test data num: {}!'.format(test_index, test_data_num))
        return test_loader

    def init_optimizer(self, **kwargs):
        model = get_valid('model', kwargs)
        lr = get('lr', kwargs, 1)
        return optim.Adadelta(model.parameters(), lr=lr)

    def init_criterion(self, **kwargs):
        super(SarRecognitionTemplate, self).init_criterion(**kwargs)
        return F.cross_entropy

    @time_consume
    def train_model(self, **kwargs):
        model = get_valid('model', kwargs)
        model.train()
        optimizer = get_valid('optimizer', kwargs)
        epoch = get_valid('epoch', kwargs)
        criterion = get_valid('criterion', kwargs)
        train_loader = get_valid('train_loader', kwargs)
        total_loss = 0

        iter_num = len(train_loader)
        log_step = iter_num // 100
        log_step = max(1, log_step)

        if iter_num == 0:
            raise RuntimeError('training data num < batch num!')
        for idx, (image, label, mask) in enumerate(train_loader):
            target = self.encode(label, False)
            target = torch.from_numpy(target)
            target = target.long()

            target_cp = self.encode(label, True)
            target_cp = torch.from_numpy(target_cp)
            target_cp = target_cp.long()

            if self.use_gpu:
                image = image.cuda()
                mask = mask.cuda()
                target = target.cuda()
                target_cp = target_cp.cuda()

            output = model(image=image, target=target, mask=mask)
            output = output.contiguous().view(-1, self.class_num)
            target_cp = target_cp[:, 1:].contiguous().view(-1)

            loss = criterion(output, target_cp, ignore_index=-1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.data.item()

            if idx % log_step == 0 and idx != 0:
                finish_percent = int(idx / iter_num * 100)
                print('Train: finish {}%, loss: {}'.format(finish_percent, loss.data.item()))

        avg_loss = total_loss / iter_num
        log = generate_log(epoch=epoch,
                           name='Train',
                           avg_loss=avg_loss)
        return log, avg_loss

    @staticmethod
    def compute_acc(pred_list, label_list):
        assert len(pred_list) == len(label_list)
        total = len(pred_list)
        correct = 0
        for idx in range(total):
            pred = pred_list[idx]
            label = label_list[idx]
            if pred == label:
                correct += 1
        return correct / total, correct, total

    @time_consume
    def test_model(self, **kwargs):
        model = get_valid('model', kwargs)
        model.eval()
        epoch = get_valid('epoch', kwargs)
        criterion = get_valid('criterion', kwargs)
        test_loader = get_valid('test_loader', kwargs)
        total_loss = 0

        iter_num = len(test_loader)
        log_step = iter_num // 100
        log_step = max(1, log_step)

        total_correct = 0
        total_pred = 0

        with torch.no_grad():
            for idx, (image, label, mask) in enumerate(test_loader):
                target = self.encode(label, False)
                target = torch.from_numpy(target)
                target = target.long()

                target_cp = self.encode(label, True)
                target_cp = torch.from_numpy(target_cp)
                target_cp = target_cp.long()

                if self.use_gpu:
                    image = image.cuda()
                    target = target.cuda()
                    mask = mask.cuda()
                    target_cp = target_cp.cuda()

                output = model(image=image, target=target, mask=mask)

                pred_label = self.decode(output)
                acc, correct, pred_num = self.compute_acc(pred_label, label)

                total_correct += correct
                total_pred += pred_num

                output = output.contiguous().view(-1, self.class_num)
                target_cp = target_cp[:, 1:].contiguous().view(-1)

                loss = criterion(output, target_cp, ignore_index=-1)
                total_loss += loss.data.item()

                if idx % log_step == 0 and idx != 0:
                    finish_percent = int(idx / iter_num * 100)
                    print('Test: finish {}%, loss: {}'.format(finish_percent, loss.data.item()))

        acc = total_correct / total_pred
        avg_loss = total_loss / iter_num
        log = generate_log(epoch=epoch,
                           name='Test',
                           avg_loss=avg_loss)
        return log, avg_loss, acc

    @time_consume
    def eval_model(self, **kwargs):
        super(SarRecognitionTemplate, self).eval_model(**kwargs)

    def load_model(self, **kwargs):
        return super(SarRecognitionTemplate, self).load_model(**kwargs)

    def save_model(self, **kwargs):
        return super(SarRecognitionTemplate, self).save_model(**kwargs)

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
            model, start_epoch, best_acc, min_loss = self.load_model(model=model, **kwargs)
        else:
            start_epoch = 0
            best_acc = 0
            min_loss = 999999

        retrain = get('retrain', kwargs, False)
        if retrain is True:
            best_acc = 0
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
                if epoch % test_step == 0 and epoch != 0:
                    self.save_model(model=model,
                                    ckpt_folder=ckpt_folder,
                                    epoch=epoch,
                                    loss=train_loss
                                    )
            else:
                if epoch % test_step == 0 and epoch != 0:
                    test_log, test_loss, acc = self.test_model(test_loader=test_loader,
                                                               criterion=criterion,
                                                               model=model,
                                                               epoch=epoch,
                                                               **kwargs)
                    if save_by_loss:
                        if test_loss < min_loss:
                            min_loss = test_loss
                            self.save_model(model=model,
                                            ckpt_folder=ckpt_folder,
                                            epoch=epoch,
                                            loss=test_loss,
                                            eval_type='accuracy',
                                            eval_score=acc)
                    else:
                        if acc >= best_acc and acc != 0:
                            best_acc = acc
                            self.save_model(model=model,
                                            ckpt_folder=ckpt_folder,
                                            epoch=epoch,
                                            loss=test_loss,
                                            eval_type='accuracy',
                                            eval_score=acc)

                    test_log += 'Eval: acc: {}\n'.format(acc)
                    log = open(log_path, 'a', encoding=self.encoding)
                    log.write(test_log)
                    log.close()
                    print(test_log)


class MultiDecoderSarRecognitionTemplate(SarRecognitionTemplate):
    def __init__(self, **kwargs):
        self.use_gpu = get('use_gpu', kwargs, True)
        if self.use_gpu:
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA is not availabel!')
            self.gpu = self.generate_gpu()
        else:
            if torch.cuda.is_available():
                print('CUDA is availabel but not use it!')
            self.gpu = None
        self.encoding = get('encoding', kwargs, 'utf-8')

        word_index_list = get_valid('word_index_list', kwargs)
        self.word2num_list = list()
        self.num2word_list = list()
        self.class_num_list = list()
        for word_index_path in word_index_list:
            word2num, num2word = self.generate_dict(word_index_path)
            assert len(word2num) == len(num2word)
            self.word2num_list.append(word2num)
            self.num2word_list.append(num2word)
            self.class_num_list.append(len(word2num))
        self.max_len = get('max_len', kwargs, 64)

    def generate_gpu(self):
        gpu_num = torch.cuda.device_count()
        gpu = list()
        for idx in range(gpu_num):
            gpu.append(idx)
        return gpu

    def encode(self, line_list, ignore_index, cls):
        word2num = self.word2num_list[cls]

        mx = 0
        num = len(line_list)
        new_line_list = list()
        for line in line_list:
            l = len(line)
            if l > self.max_len - 2:
                line = line[0:self.max_len - 2]
                mx = self.max_len - 2
            if l > mx:
                mx = l
            new_line_list.append(line)

        label = np.zeros((num, mx + 2), dtype=np.int32)
        if ignore_index:
            label -= 1

        row = 0
        for line in new_line_list:
            line = line.strip()
            line = line.replace(' ', '')
            col = 1
            for word in line:
                l = word2num[word]
                label[row, col] = l
                col += 1
            label[row, 0] = 0
            label[row, col] = 0
            row += 1
        return label

    def decode(self, preds_tensor, cls):
        num2word = self.num2word_list[cls]

        label_list = list()
        for pred in preds_tensor:
            pred = pred.cpu().numpy()
            label = ''
            for p in pred:
                p = int(p.argmax())
                if p != 0:
                    label += num2word[p]
                else:
                    break
            label_list.append(label)
        return label_list

    def init_model(self, **kwargs):
        model_factory = RecognitionModelFactory()
        model = model_factory.get_model(class_num_list=self.class_num_list, **kwargs)
        return model

    def init_criterion(self, **kwargs):
        criterion = dict()
        criterion['sar'] = F.cross_entropy
        criterion['cls'] = torch.nn.CrossEntropyLoss()

        return criterion

    def init_trainloader(self, **kwargs):
        train_transforms = get('train_transforms', kwargs, None)
        if train_transforms is None:
            distort_num = get('distort_num', kwargs, 10)
            distort_ratio = get('distort_ratio', kwargs, 0.1)
            train_transforms = transforms.Compose([
                RandomDistort(distort_num, distort_ratio),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        kwargs['transforms'] = train_transforms
        train_folder = get_valid('train_folder', kwargs)
        class_dict = get_valid('class_dict', kwargs)
        train_batch = get('train_batch', kwargs, 64)
        train_worker = get('train_worker', kwargs, 8)
        drop_last = get('drop_last', kwargs, True)
        shuffle = get('shuffle', kwargs, True)

        train_loader_dict = dict()
        max_data_num = 0

        for cls in class_dict:
            cls_folder = os.path.join(train_folder, cls)
            cls_index_path = os.path.join(cls_folder, 'label.txt')

            train_dataset = SarDataset(index_path=cls_index_path, folder=cls_folder, **kwargs)

            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=self.get_batch_size(train_batch),
                                                       num_workers=train_worker,
                                                       drop_last=drop_last,
                                                       shuffle=shuffle)
            cls_int = class_dict[cls]
            train_loader_dict[cls_int] = dict()
            train_loader_dict[cls_int]['data_loader'] = train_loader
            data_num = len(train_dataset)
            train_loader_dict[cls_int]['data_num'] = data_num
            if data_num > max_data_num:
                max_data_num = data_num

            print('train {} data num: {}!'.format(cls, data_num))
        for cls in class_dict:
            cls_int = class_dict[cls]
            data_num = train_loader_dict[cls_int]['data_num']
            train_loader_dict[cls_int]['repeat_num'] = int(max_data_num / data_num)
        return train_loader_dict

    def init_testloader(self, **kwargs):
        test_transforms = get('test_transforms', kwargs, None)
        if test_transforms is None:
            test_transforms = transforms.Compose([
                # Denoise(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        kwargs['transforms'] = test_transforms
        test_folder = get_valid('test_folder', kwargs)
        test_batch = get('test_batch', kwargs, 64)
        test_worker = get('test_worker', kwargs, 8)

        class_dict = get_valid('class_dict', kwargs)
        test_loader_dict = dict()

        for cls in class_dict:
            cls_folder = os.path.join(test_folder, cls)
            cls_index_path = os.path.join(cls_folder, 'label.txt')

            test_dataset = SarDataset(index_path=cls_index_path, folder=cls_folder, **kwargs)

            test_loader = torch.utils.data.DataLoader(test_dataset,
                                                      batch_size=self.get_batch_size(test_batch),
                                                      num_workers=test_worker)
            cls_int = class_dict[cls]
            test_loader_dict[cls_int] = test_loader

            print('test {} data num: {}!'.format(cls, len(test_dataset)))
        return test_loader_dict

    @time_consume
    def train_model(self, **kwargs):
        model = get_valid('model', kwargs)
        model.train()
        optimizer = get_valid('optimizer', kwargs)
        epoch = get_valid('epoch', kwargs)
        criterion = get_valid('criterion', kwargs)
        train_loader = get_valid('train_loader', kwargs)
        total_loss = 0

        class_dict = get_valid('class_dict', kwargs)
        key_list = list()
        for key in class_dict:
            key_list.append(key)
        cls = random.randint(0, len(key_list) - 1)
        cls = key_list[cls]
        cls_label = class_dict[cls]

        data_loader = train_loader[cls_label]['data_loader']
        repeat_num = train_loader[cls_label]['repeat_num']

        iter_num = len(data_loader)
        log_step = iter_num // 100
        log_step = max(1, log_step)

        model.fc.weight.requires_grad = False
        model.encoder.requires_grad = False

        if iter_num == 0:
            raise RuntimeError('training data num < batch num!')

        for idy in range(repeat_num):
            for idx, (image, label, mask) in enumerate(data_loader):
                target = self.encode(label, False, cls_label)
                target = torch.from_numpy(target)
                target = target.long()

                target_cp = self.encode(label, True, cls_label)
                target_cp = torch.from_numpy(target_cp)
                target_cp = target_cp.long()

                batch_size = image.shape[0]
                cls_tensor = torch.zeros((batch_size))
                cls_tensor += cls_label
                cls_tensor = cls_tensor.long()

                if self.use_gpu:
                    image = image.cuda()
                    mask = mask.cuda()
                    target = target.cuda()
                    target_cp = target_cp.cuda()

                    cls_tensor = cls_tensor.cuda()

                output, cls_pred = model(image=image, target=target, mask=mask, cls_label=cls_label, type='sar')

                class_num = self.class_num_list[cls_label]

                output = output.contiguous().view(-1, class_num)
                target_cp = target_cp[:, 1:].contiguous().view(-1)

                sar_loss = criterion['sar'](output, target_cp, ignore_index=-1)
                cls_loss = criterion['cls'](cls_pred, cls_tensor)
                loss = sar_loss + cls_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.data.item()

                if idx % log_step == 0 and idx != 0:
                    finish_percent = int(idx / iter_num * 100)
                    print('Train {}: {}/{} finish {}%, loss: {}, sar loss:{}, cls loss: {}'.format(cls,
                                                                                                   idy,
                                                                                                   repeat_num,
                                                                                                   finish_percent,
                                                                                                   loss.data.item(),
                                                                                                   sar_loss.item(),
                                                                                                   cls_loss.item()))

        avg_loss = total_loss / iter_num
        log = generate_log(epoch=epoch,
                           name='Train',
                           avg_loss=avg_loss)

        return log, avg_loss

    @time_consume
    def test_model(self, **kwargs):
        model = get_valid('model', kwargs)
        model.eval()
        epoch = get_valid('epoch', kwargs)
        criterion = get_valid('criterion', kwargs)
        test_loader = get_valid('test_loader', kwargs)
        total_loss = 0

        total_correct = 0
        total_pred = 0

        class_dict = get_valid('class_dict', kwargs)

        for cls in class_dict:
            cls_label = class_dict[cls]
            data_loader = test_loader[cls_label]
            iter_num = len(data_loader)
            log_step = iter_num // 100
            log_step = max(1, log_step)

            class_correct = 0
            class_pred = 0

            with torch.no_grad():
                for idx, (image, label, mask) in enumerate(data_loader):
                    target = self.encode(label, False, cls_label)
                    target = torch.from_numpy(target)
                    target = target.long()

                    target_cp = self.encode(label, True, cls_label)
                    target_cp = torch.from_numpy(target_cp)
                    target_cp = target_cp.long()

                    batch_size = image.shape[0]
                    cls_tensor = torch.zeros((batch_size))
                    cls_tensor += cls_label
                    cls_tensor = cls_tensor.long()

                    if self.use_gpu:
                        image = image.cuda()
                        target = target.cuda()
                        mask = mask.cuda()
                        target_cp = target_cp.cuda()
                        cls_tensor = cls_tensor.cuda()

                    output, cls_pred = model(image=image, target=target, mask=mask, cls_label=cls_label, type='sar')
                    pred_label = self.decode(output, cls_label)

                    acc, correct, pred_num = self.compute_acc(pred_label, label)

                    total_correct += correct
                    total_pred += pred_num

                    class_correct += correct
                    class_pred += pred_num

                    class_num = self.class_num_list[cls_label]
                    output = output.contiguous().view(-1, class_num)
                    target_cp = target_cp[:, 1:].contiguous().view(-1)

                    sar_loss = criterion['sar'](output, target_cp, ignore_index=-1)
                    cls_loss = criterion['cls'](cls_pred, cls_tensor)
                    loss = sar_loss + cls_loss

                    total_loss += loss.data.item()

                    if idx % log_step == 0 and idx != 0:
                        finish_percent = int(idx / iter_num * 100)
                        print('Test {}: finish {}%, loss: {}, sar loss:{}, cls loss: {}'.format(cls,
                                                                                                finish_percent,
                                                                                                loss.data.item(),
                                                                                                sar_loss.item(),
                                                                                                cls_loss.item()))
                print('Test {}: acc: {}'.format(cls, class_correct / class_pred))

        acc = total_correct / total_pred
        avg_loss = total_loss / iter_num
        log = generate_log(epoch=epoch,
                           name='Test',
                           avg_loss=avg_loss)
        return log, avg_loss, acc

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
            model, start_epoch, best_acc, min_loss = self.load_model(model=model, **kwargs)
        else:
            start_epoch = 0
            best_acc = 0
            min_loss = 999999

        if self.gpu is not None:
            model.cuda()

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
                if epoch % test_step == 0 and epoch != 0:
                    self.save_model(model=model,
                                    ckpt_folder=ckpt_folder,
                                    epoch=epoch,
                                    loss=train_loss
                                    )
            else:
                if epoch % test_step == 0 and epoch != 0:
                    test_log, test_loss, acc = self.test_model(test_loader=test_loader,
                                                               criterion=criterion,
                                                               model=model,
                                                               epoch=epoch,
                                                               **kwargs)
                    if save_by_loss:
                        if test_loss < min_loss:
                            min_loss = test_loss
                            self.save_model(model=model,
                                            ckpt_folder=ckpt_folder,
                                            epoch=epoch,
                                            loss=test_loss,
                                            eval_type='accuracy',
                                            eval_score=acc)
                    else:
                        if acc > best_acc:
                            best_acc = acc
                            self.save_model(model=model,
                                            ckpt_folder=ckpt_folder,
                                            epoch=epoch,
                                            loss=test_loss,
                                            eval_type='accuracy',
                                            eval_score=acc)

                    test_log += 'Eval: acc: {}\n'.format(acc)
                    log = open(log_path, 'a', encoding=self.encoding)
                    log.write(test_log)
                    log.close()
                    print(test_log)


class MultiDecoderClassifyTemplate(SarRecognitionTemplate):
    def __init__(self, **kwargs):
        self.use_gpu = get('use_gpu', kwargs, True)
        if self.use_gpu:
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA is not availabel!')
            self.gpu = self.generate_gpu()
        else:
            if torch.cuda.is_available():
                print('CUDA is availabel but not use it!')
            self.gpu = None
        self.encoding = get('encoding', kwargs, 'utf-8')

        word_index_list = get_valid('word_index_list', kwargs)
        self.word2num_list = list()
        self.num2word_list = list()
        self.class_num_list = list()
        for word_index_path in word_index_list:
            word2num, num2word = self.generate_dict(word_index_path)
            assert len(word2num) == len(num2word)
            self.word2num_list.append(word2num)
            self.num2word_list.append(num2word)
            self.class_num_list.append(len(word2num))
        self.max_len = get('max_len', kwargs, 64)

    def generate_gpu(self):
        gpu_num = torch.cuda.device_count()
        gpu = list()
        for idx in range(gpu_num):
            gpu.append(idx)
        return gpu

    def encode(self, line_list, ignore_index, cls):
        word2num = self.word2num_list[cls]

        mx = 0
        num = len(line_list)
        new_line_list = list()
        for line in line_list:
            l = len(line)
            if l > self.max_len - 2:
                line = line[0:self.max_len - 2]
                mx = self.max_len - 2
            if l > mx:
                mx = l
            new_line_list.append(line)

        label = np.zeros((num, mx + 2), dtype=np.int32)
        if ignore_index:
            label -= 1

        row = 0
        for line in new_line_list:
            line = line.strip()
            line = line.replace(' ', '')
            col = 1
            for word in line:
                l = word2num[word]
                label[row, col] = l
                col += 1
            label[row, 0] = 0
            label[row, col] = 0
            row += 1
        return label

    def decode(self, preds_tensor, cls):
        num2word = self.num2word_list[cls]

        label_list = list()
        for pred in preds_tensor:
            pred = pred.cpu().numpy()
            label = ''
            for p in pred:
                p = int(p.argmax())
                if p != 0:
                    label += num2word[p]
                else:
                    break
            label_list.append(label)
        return label_list

    def init_model(self, **kwargs):
        model_factory = RecognitionModelFactory()
        model = model_factory.get_model(class_num_list=self.class_num_list, **kwargs)
        return model

    def init_criterion(self, **kwargs):
        criterion = dict()
        criterion['sar'] = F.cross_entropy
        criterion['cls'] = torch.nn.CrossEntropyLoss()

        return criterion

    def init_trainloader(self, **kwargs):
        train_transforms = get('train_transforms', kwargs, None)
        if train_transforms is None:
            distort_num = get('distort_num', kwargs, 10)
            distort_ratio = get('distort_ratio', kwargs, 0.1)
            train_transforms = transforms.Compose([
                RandomDistort(distort_num, distort_ratio),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        train_folder = get_valid('train_folder', kwargs)
        train_dataset = MultiDecoderClassifyDataset(folder=train_folder,
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
        test_transforms = get('test_transforms', kwargs, None)
        if test_transforms is None:
            test_transforms = transforms.Compose([
                Denoise(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        test_folder = get_valid('test_folder', kwargs)

        test_dataset = MultiDecoderClassifyDataset(folder=test_folder,
                                                   is_training=False,
                                                   transforms=test_transforms,
                                                   **kwargs)

        test_batch = get('test_batch', kwargs, 16)
        test_worker = get('test_worker', kwargs, 8)

        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=test_batch,
                                                  num_workers=test_worker)
        test_data_num = len(test_dataset)
        self.test_data_num = test_data_num
        print('generate test data loader, test data folder: {}, test data num:{}'.format(test_folder,
                                                                                         test_data_num))
        return test_loader

    @time_consume
    def train_model(self, **kwargs):
        model = get_valid('model', kwargs)
        model.train()
        optimizer = get_valid('optimizer', kwargs)
        epoch = get_valid('epoch', kwargs)
        criterion = get_valid('criterion', kwargs)
        train_loader = get_valid('train_loader', kwargs)
        total_loss = 0

        iter_num = len(train_loader)
        log_step = iter_num // 100
        log_step = max(1, log_step)

        for idx, data in enumerate(train_loader):
            img, label = data[0], data[1]
            if self.gpu is not None:
                img = img.cuda()
                label = label.cuda()
                model = model.cuda()

            optimizer.zero_grad()
            pred = model(image=img, type='classify')
            loss = criterion['cls'](pred, label)
            loss.backward()
            optimizer.step()

            if idx % log_step == 0 and idx != 0:
                finish_percent = int(idx / iter_num * 100)
                print('Train: finish {}%, loss: {}'.format(finish_percent, loss.data.item()))
            total_loss += loss.data.item()

        avg_loss = total_loss / iter_num
        log = generate_log(epoch=epoch,
                           name='Train',
                           avg_loss=avg_loss)
        return log, avg_loss

    @time_consume
    def test_model(self, **kwargs):
        model = get_valid('model', kwargs)
        model.eval()
        epoch = get_valid('epoch', kwargs)
        criterion = get_valid('criterion', kwargs)
        test_loader = get_valid('test_loader', kwargs)
        total_loss = 0

        total_correct = 0

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
                pred = model(image=img, type='classify')
                loss = criterion['cls'](pred, label)
                _, pred_cls = torch.max(pred, 1)
                total_correct += torch.sum(pred_cls.data == label.data)
                total_loss += loss.data.item()

                if idx % log_step == 0 and idx != 0:
                    finish_percent = int(idx / iter_num * 100)
                    print('Test: finish {}%, loss: {}'.format(finish_percent, loss.data.item()))

        acc = int(total_correct) / self.test_data_num
        avg_loss = total_loss / iter_num
        log = generate_log(epoch=epoch,
                           name='Test',
                           avg_loss=avg_loss)
        return log, avg_loss, acc

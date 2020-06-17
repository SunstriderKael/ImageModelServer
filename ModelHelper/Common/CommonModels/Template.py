from ModelHelper.Common.CommonUtils import get, get_valid
from abc import abstractmethod, ABCMeta
import torch
import torch.nn as nn
import os
from collections import OrderedDict


class AbstractTemplate(metaclass=ABCMeta):
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

    def generate_gpu(self):
        gpu_num = torch.cuda.device_count()
        gpu = list()
        for idx in range(gpu_num):
            gpu.append(idx)
        return gpu

    def get_batch_size(self, batch_per_gpu):
        if self.gpu is None:
            return batch_per_gpu
        else:
            return len(self.gpu) * batch_per_gpu

    @abstractmethod
    def init_model(self, **kwargs):
        model_factory = get_valid('model_factory', kwargs)
        model = model_factory.get_model(**kwargs)
        return model

    @abstractmethod
    def init_trainloader(self, **kwargs):
        pass

    @abstractmethod
    def init_testloader(self, **kwargs):
        pass

    @abstractmethod
    def init_optimizer(self, **kwargs):
        pass

    @abstractmethod
    def init_criterion(self, **kwargs):
        pass

    @abstractmethod
    def train_model(self, **kwargs):
        pass

    @abstractmethod
    def test_model(self, **kwargs):
        pass

    @abstractmethod
    def eval_model(self, **kwargs):
        pass

    @abstractmethod
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
        eval_score = ckpt['eval_score']
        return model, start_epoch, eval_score, loss

    @abstractmethod
    def save_model(self, **kwargs):
        model = get_valid('model', kwargs)
        ckpt_folder = get_valid('ckpt_folder', kwargs)
        epoch = get_valid('epoch', kwargs)
        loss = get_valid('loss', kwargs)
        eval_type = get('eval_type', kwargs, 'score')
        eval_score = get('eval_score', kwargs, 0)
        model_name = get('model_name', kwargs, '')

        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'eval_type': eval_type,
            'eval_score': eval_score,
            'loss': loss
        }

        if eval_type is None:
            eval_type = ''
        if eval_score is None:
            eval_score = ''
        else:
            eval_score = round(eval_score, 4)
            eval_score = str(eval_score)
        loss = round(loss, 4)
        loss = str(loss)

        ckpt_name = '{}epoch{}{}{}loss{}.pth'.format(model_name, epoch, eval_type, eval_score, loss)
        ckpt_path = os.path.join(ckpt_folder, ckpt_name)
        torch.save(state, ckpt_path)
        print('Save checkpoint: {}'.format(ckpt_path))
        return ckpt_path

    @abstractmethod
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
            model, start_epoch, max_score, min_loss = self.load_model(model=model, **kwargs)
        else:
            start_epoch = 0
            max_score = 0
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
                                                     epoch=epoch)
            log = open(log_path, 'a', encoding=self.encoding)
            log.write(train_log)
            log.close()
            print(train_log)

            if not do_test:
                if epoch % test_step == 0 and epoch != 0:
                    self.save_model(model=model,
                                    ckpt_folder=ckpt_folder,
                                    epoch=epoch,
                                    loss=train_loss,
                                    **kwargs
                                    )
            else:
                if epoch % test_step == 0 and epoch != 0:
                    test_log, test_loss = self.test_model(test_loader=test_loader,
                                                          criterion=criterion,
                                                          model=model,
                                                          epoch=epoch)
                    if save_by_loss:
                        if test_loss < min_loss:
                            min_loss = test_loss
                            self.save_model(model=model,
                                            ckpt_folder=ckpt_folder,
                                            epoch=epoch,
                                            loss=test_loss,
                                            **kwargs)
                    else:
                        pred_folder = os.path.join(output_folder, 'pred')
                        if not os.path.exists(pred_folder):
                            os.makedirs(pred_folder)
                        eval_score = self.eval_model(model=model, pred_folder=pred_folder, **kwargs)
                        test_log += 'Eval: f1score: {}\n'.format(eval_score)
                        if eval_score >= max_score:
                            max_score = eval_score
                            self.save_model(model=model,
                                            ckpt_folder=ckpt_folder,
                                            epoch=epoch,
                                            loss=test_loss,
                                            eval_score=eval_score,
                                            **kwargs)

                    log = open(log_path, 'a', encoding=self.encoding)
                    log.write(test_log)
                    log.close()
                    print(test_log)

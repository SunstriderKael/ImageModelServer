import random
from multiprocessing import Pool
import hashlib
import os
import shutil


def get(key, kwargs, default=None):
    if key in kwargs:
        return kwargs[key]
    else:
        return default


def get_valid(key, kwargs):
    if key not in kwargs:
        raise RuntimeError('has no {} key in kwargs!'.format(key))
    return kwargs[key]


def generate_log(epoch, name='', **kwargs):
    log = '{}: {}th epoch'.format(name, epoch)
    for key in kwargs:
        log_item = ', {}: {}'.format(key, kwargs[key])
        log += log_item
    log += '\n'
    return log


def handle_data_multiprocess(**kwargs):
    data_list = get_valid('data_list', kwargs)
    func = get_valid('func', kwargs)
    worker_num = get_valid('worker_num', kwargs)

    total_list = list()
    p = Pool(worker_num)

    for idx in range(worker_num):
        process_list = list()
        total_list.append(process_list)

    for data in data_list:
        rand = random.randint(0, worker_num - 1)
        total_list[rand].append(data)

    for idx in range(worker_num):
        p.apply(func, args=(total_list[rand]))
    p.close()
    p.join()


def str2md5(str):
    m = hashlib.md5()
    str = str.encode(encoding='utf-8')
    m.update(str)
    return m.hexdigest()


def file_name2md5(file_name):
    ext = os.path.splitext(file_name)[-1]
    file_name = file_name.replace(ext, '')
    return str2md5(file_name)+ext


def file_name2md5_infolder(src_folder, desc_folder):
    if not os.path.exists(desc_folder):
        os.makedirs(desc_folder)
    for root, dirs, files in os.walk(src_folder):
        for file in files:
            src_path = os.path.join(root, file)
            folder = root.replace(src_folder, desc_folder)
            if not os.path.exists(folder):
                os.makedirs(folder)
            desc_path = os.path.join(folder, file_name2md5(file))
            shutil.copy(src_path, desc_path)
            print('copy {} to {}'.format(src_path, desc_path))


if __name__ == '__main__':
    src_folder = '/home/gaoyuanzi/Documents/tmp/test1'
    desc_folder = '/home/gaoyuanzi/Documents/tmp/output'
    file_name2md5_infolder(src_folder, desc_folder)

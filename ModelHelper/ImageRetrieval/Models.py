from ModelHelper.Common.CommonUtils import get, get_valid
import torch
from torchvision import transforms
import os
from PIL import Image
from ModelHelper.ImageRetrieval.lshash.lshash import LSHash
from ModelHelper.ImageRetrieval.cirtorch.networks.imageretrievalnet import init_network, extract_vectors
from ModelHelper.ImageRetrieval.cirtorch.datasets.datahelpers import default_loader, imresize


class ImageProcess:
    def __init__(self, img_dir):
        self.img_dir = img_dir

    def process(self):
        imgs = list()
        for root, dirs, files in os.walk(self.img_dir):
            for file in files:
                img_path = os.path.join(root + os.sep, file)
                try:
                    image = Image.open(img_path)
                    if max(image.size) / min(image.size) < 5:
                        imgs.append(img_path)
                    else:
                        continue
                except:
                    print("image height/width ratio is small")

        return imgs


class LSHRetrieval:
    def __init__(self, **kwargs):
        self.checkpoint = get_valid('checkpoint', kwargs)
        self.state = torch.load(self.checkpoint)
        net_params = self.init_params()
        self.model = init_network(net_params)
        self.model.load_state_dict(self.state['state_dict'])
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()
        multiscale = '[1]'
        self.ms = list(eval(multiscale))

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.model.meta['mean'],
                std=self.model.meta['std']
            )
        ])

        self.hash_size = get('hash_size', kwargs, 0)
        self.num_hashtables = get('num_hashtables', kwargs, 1)
        self.input_dim = get('input_dim', kwargs, 2048)
        self.image_size = get('image_size', kwargs, 1024)

    def init_params(self):
        net_params = {}
        net_params['architecture'] = self.state['meta']['architecture']
        net_params['pooling'] = self.state['meta']['pooling']
        net_params['local_whitening'] = self.state['meta'].get('local_whitening', False)
        net_params['regional'] = self.state['meta'].get('regional', False)
        net_params['whitening'] = self.state['meta'].get('whitening', False)
        net_params['mean'] = self.state['meta']['mean']
        net_params['std'] = self.state['meta']['std']
        net_params['pretrained'] = False

        return net_params

    def get_lsh(self, img_library_folder):
        assert os.path.exists(img_library_folder)

        images = ImageProcess(img_library_folder).process()
        print('extract library feature')
        vecs, img_paths = extract_vectors(self.model, images, 1024, self.transform, ms=self.ms)
        feature_dict = dict(zip(img_paths, list(vecs.detach().cpu().numpy().T)))
        lsh = LSHash(hash_size=int(self.hash_size), input_dim=int(self.input_dim),
                     num_hashtables=int(self.num_hashtables))
        for img_path, vec in feature_dict.items():
            lsh.index(vec.flatten(), extra_data=img_path)
        return lsh

    def get_initial_lsh(self):
        return LSHash(hash_size=int(self.hash_size), input_dim=int(self.input_dim),
                      num_hashtables=int(self.num_hashtables))

    def extract_feature(self, img):
        # img = Image.open(img_path)
        img = imresize(img, self.image_size)
        img = self.transform(img)
        input = img.unsqueeze(0)
        with torch.no_grad():
            if torch.cuda.is_available():
                input = input.cuda()
            feature = self.model(input).cpu().data.squeeze()

        return feature

    def retrieval(self, **kwargs):
        target_folder = get_valid('target_folder', kwargs)
        lsh = get_valid('lsh', kwargs)
        query_num = get('query_num', kwargs, 1)
        query_num = int(query_num)

        images = ImageProcess(target_folder).process()
        print('extract target feature')
        vecs, img_paths = extract_vectors(self.model, images, 1024, self.transform, ms=self.ms)
        target_feature = dict(zip(img_paths, list(vecs.detach().cpu().numpy().T)))

        for q_path, q_vec in target_feature.items():
            try:
                response = lsh.query(q_vec.flatten(), num_results=query_num, distance_func="cosine")
                print('target img: {}'.format(q_path))
                for idx in range(query_num):
                    query_img_path = response[idx][0][1]
                    print('{}th query img: {}'.format(idx, query_img_path))
                print('*' * 20)
            except:
                print('error occur on: {}'.format(q_path))
                continue

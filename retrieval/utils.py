import numpy as np
import json
import torch


def feature2str(feature):
    feature = feature.numpy()
    feature = feature.tolist()
    feature = json.dumps(feature)
    return feature


def str2feature(feature):
    feature = json.loads(feature)
    feature = np.array(feature)
    feature = torch.from_numpy(feature)
    return feature

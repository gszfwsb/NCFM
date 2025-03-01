import numpy as np


def subsample(data, target, max_size=-1):
    if (data.shape[0] > max_size) and (max_size > 0):
        indices = np.random.permutation(data.shape[0])
        data = data[indices[:max_size]]
        target = target[indices[:max_size]]

    return data, target

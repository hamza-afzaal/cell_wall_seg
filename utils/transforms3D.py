import numpy as np
import torch
import random
from skimage.filters import gaussian
from skimage.segmentation import find_boundaries
import elasticdeform

RAND_STATE = np.random.RandomState(45)


class GaussianBlur3D:
    def __init__(self, sigma=1, truncate=3, multichannel=False):
        self.sigma = sigma
        self.truncate = truncate
        self.multichannel = multichannel

    def __call__(self, m):
        return np.where(gaussian(m, sigma=self.sigma,
                                 truncate=self.truncate, multichannel=self.multichannel) > 0.3, 1, 0)


class GroundTruthToBoundary3D:
    def __init__(self, connectivity=2, mode='thick'):
        self.connectivity = connectivity
        self.mode = mode

    def __call__(self, m):
        return find_boundaries(m, connectivity=2, mode='thick')


class OutputClassLabelling:
    def __init__(self, ground_label=False):
        self.ground_label = ground_label

    def __call__(self, m):
        if m is not None and type(m) is tuple:
            m, ground_labels = m[0], m[1]
        else:
            ground_labels = {}

        boundary_label = np.where(m > 0.35, 1, 0)
        background_label = np.where(boundary_label > 0, 0, 1)

        if self.ground_label:
            ground_labels = dict({'boundary_label': boundary_label, 'background_label': background_label},
                                 # 'background_label': background_label
                                 **ground_labels)
        else:
            ground_labels = {'boundary_label': boundary_label, 'background_label': background_label}
            # , 'background_label': background_label

        return m, ground_labels


class Normalize3D:
    def __init__(self, dict_key='normalized', standardize=True):
        self.standardize = standardize
        self.dict_key = dict_key

    def __call__(self, m):
        normalized = m

        if np.max(m) != 0:
            normalized = (m - np.min(m)) / np.max(m)
            if self.standardize:
                normalized = (normalized - normalized.mean()) / normalized.std()

        return normalized


class ToTensor3D:
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, m):
        return torch.from_numpy(m).type(self.dtype)


class ElasticDeformation3D:
    def __init__(self, threshold=0.5, sigma=4, order=None):
        if order is None:
            order = [3, 0]
        self.sigma = sigma
        self.order = order
        self.threshold = threshold

    def __call__(self, m):
        if random.random() > 0.5:
            x_dtype = m['x'].dtype
            y_dtype = m['y'].dtype

            data = [m['x'].numpy(), m['y'].numpy()]

            deforms = elasticdeform.deform_random_grid(data, sigma=self.sigma, axis=[(2, 3), (2, 3)], order=self.order)
            m['x'] = torch.tensor(deforms[0], dtype=x_dtype)
            m['y'] = torch.tensor(np.where(deforms[1] > 0.5, 1, 0), dtype=y_dtype)
        if torch.max(m['x']) != 0:
            m['x'] = (m['x'] - m['x'].mean()) / m['x'].std()
        return m


class RandomFlip3D:
    def __init__(self, threshold=0.5, orientation='horizontal'):
        self.threshold = threshold
        self.orientation = 2 if orientation == 'horizontal' else 3

    def __call__(self, m):
        if random.random() > 0.5:
            for key in m:
                dtype = m[key].dtype
                m[key] = torch.tensor(np.ascontiguousarray(np.flip(m[key].numpy(), axis=self.orientation)),
                                      dtype=dtype)
        return m


class AdditiveGaussianNoise:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, m):
        if random.random() > 0.5:
            std_val = RAND_STATE.uniform(0, 1)
            noise = RAND_STATE.normal(0, std_val, size=m['x'].shape).astype('f')
            m['x'] = m['x'] + noise
        return m


class AdditivePoissonNoise:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, m):
        if random.random() > 0.5:
            std_val = RAND_STATE.uniform(0, 1)
            noise = RAND_STATE.poisson(std_val, size=m['x'].shape).astype('f')
            m['x'] = m['x'] + noise
        return m


class ToNumpy:
    def __call__(self, m):
        return m.numpy()

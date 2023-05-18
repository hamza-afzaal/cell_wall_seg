import numpy as np
import torch
import random
from skimage.filters import gaussian
from skimage.segmentation import find_boundaries
import elasticdeform

RAND_STATE = np.random.RandomState(45)


class GaussianBlur:
    def __init__(self, dict_key='blurred', sigma=1, truncate=3, multichannel=False, ground_label=False):
        self.sigma = sigma
        self.truncate = truncate
        self.multichannel = multichannel
        self.ground_label = ground_label
        self.dict_key = dict_key

    def __call__(self, m):
        if m is not None and type(m) is tuple:
            m, ground_labels = m[0], m[1]
        else:
            ground_labels = {}

        blurred = gaussian(m, sigma=self.sigma, truncate=self.truncate, multichannel=self.multichannel)

        if self.ground_label:
            ground_labels = dict({self.dict_key: blurred}, **ground_labels)
        else:
            ground_labels = {self.dict_key: blurred}

        return blurred, ground_labels


class GroundTruthToBoundary:
    def __init__(self, dict_key='boundaries', connectivity=2, mode='thick', ground_label=False):
        self.connectivity = connectivity
        self.mode = mode
        self.ground_label = ground_label
        self.dict_key = dict_key

    def __call__(self, m):
        if m is not None and type(m) is tuple:
            m, ground_labels = m[0], m[1]
        else:
            ground_labels = {}

        boundaries = find_boundaries(m, connectivity=2, mode='thick')

        if self.ground_label:
            ground_labels = dict({self.dict_key: boundaries}, **ground_labels)
        else:
            ground_labels = {self.dict_key: boundaries}

        return boundaries, ground_labels


class OutputClassLabelling:
    def __init__(self, ground_label=False):
        self.ground_label = ground_label

    def __call__(self, m):
        if m is not None and type(m) is tuple:
            m, ground_labels = m[0], m[1]
        else:
            ground_labels = {}

        boundary_label = np.where(m > 0.3, 1, 0)
        background_label = np.where(boundary_label > 0, 0, 1)

        if self.ground_label:
            ground_labels = dict({'boundary_label': boundary_label, 'background_label': background_label},
                                 # 'background_label': background_label
                                 **ground_labels)
        else:
            ground_labels = {'boundary_label': boundary_label, 'background_label': background_label}
            # , 'background_label': background_label

        return m, ground_labels


class Normalize:
    def __init__(self, dict_key='normalized', ground_label=True):
        self.ground_label = ground_label
        self.dict_key = dict_key

    def __call__(self, m):
        if m is not None and type(m) is tuple:
            m, ground_labels = m[0], m[1]
        else:
            ground_labels = {}

        normalized = m
        mean = 0
        std = 0
        if np.max(m) != 0:
            normalized = (m - np.min(m)) / np.ptp(m)
            mean = normalized.mean()
            std = normalized.std()
            normalized = (normalized - mean) / std

        if self.ground_label:
            ground_labels = dict({self.dict_key: normalized, 'mean': mean, 'std': std}, **ground_labels)
        else:
            ground_labels = {self.dict_key: normalized, 'mean': mean, 'std': std}

        return normalized, ground_labels


class ToTensor:
    def __init__(self, dict_keys=None):
        self.dict_keys = dict_keys

    def __call__(self, m):
        if m is not None and type(m) is tuple:
            m, ground_labels = m[0], m[1]
        else:
            ground_labels = {}

        for key in self.dict_keys:
            if key in ground_labels.keys():
                ground_labels[key] = torch.from_numpy(ground_labels[key]).type(self.dict_keys[key])

        return ground_labels


class ElasticDeformation:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, m):
        if random.random() > 0.5:
            deforms = elasticdeform.deform_random_grid([m['x'], m['y']], sigma=25, axis=[(1, 2), (1, 2)])
            m['x'] = deforms[0]
            m['y'] = deforms[1]
        return m


class RandomFlip:
    def __init__(self, threshold=0.5, orientation='horizontal'):
        self.threshold = threshold
        self.orientation = 2 if orientation == 'horizontal' else 1

    def __call__(self, m):
        if random.random() > 0.5:
            for key in m:
                m[key] = np.ascontiguousarray(np.flip(m[key], axis=self.orientation))
        return m


class ElasticDeformationV2:
    def __init__(self, threshold=0.5, input_keys=None, output_keys=None):
        if output_keys is None:
            output_keys = {}
        if input_keys is None:
            input_keys = {}
        self.threshold = threshold
        self.input_keys = input_keys
        self.output_keys = output_keys

    def __call__(self, m):
        if random.random() > 0.5:
            data = []
            key = ['x', 'y']
            mean = m[key[0]]['mean']
            std = m[key[0]]['std']
            for values in self.input_keys:
                if mean != 0 or std != 0:
                    data.append((m[key[0]][values].numpy() * std) + mean)
                else:
                    data.append(m[key[0]][values].numpy())
            for values in self.output_keys:
                data.append(m[key[1]][values].numpy())

            deforms = elasticdeform.deform_random_grid(data, sigma=8, axis=[(1, 2), (1, 2)], order=[3, 0])
            # add spline order of 3 and 0 here

            counter = 0
            m[key[0]]['mean'] = deforms[counter].mean()
            if deforms[counter].std() > 1e-10:
                m[key[0]]['std'] = deforms[counter].std()
            else:
                m[key[0]]['std'] = 0
            for values in self.input_keys:
                if np.max(deforms[counter]) != 0 and m[key[0]]['std'] != 0:
                    m[key[0]][values] = torch.tensor(deforms[counter], dtype=self.input_keys[values])
                    # m[key[0]][values] = torch.tensor((deforms[counter] - deforms[counter].mean()) /
                    #                                  deforms[counter].std(), dtype=self.input_keys[values])
                else:
                    m[key[0]][values] = torch.tensor(deforms[counter], dtype=self.input_keys[values])
                counter += 1
            for values in self.output_keys:
                m[key[1]][values] = torch.tensor(np.where(deforms[counter] > 0.5, 1, 0), dtype=self.output_keys[values])
                counter += 1
        return m


class RandomFlipV2:
    def __init__(self, threshold=0.5, orientation='horizontal', input_keys=None, output_keys=None):
        if output_keys is None:
            output_keys = {}
        if input_keys is None:
            input_keys = {}
        self.threshold = threshold
        self.orientation = 2 if orientation == 'horizontal' else 1
        self.input_keys = input_keys
        self.output_keys = output_keys

    def __call__(self, m):
        if random.random() > 0.5:
            key = 'x'
            for values in self.input_keys:
                m[key][values] = torch.tensor(np.ascontiguousarray(np.flip(m[key][values].numpy(), axis=self.orientation)),
                                              dtype=self.input_keys[values])
            key = 'y'
            for values in self.output_keys:
                m[key][values] = torch.tensor(np.ascontiguousarray(np.flip(m[key][values].numpy(), axis=self.orientation)),
                                              dtype=self.output_keys[values])
        return m


class ToNumpy:
    def __call__(self, m):
        return m.numpy()


class AdditiveGaussianNoise:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, m):
        if random.random() > 0.5:
            std_val = RAND_STATE.uniform(0, 1)
            noise = RAND_STATE.normal(0, std_val, size=m['x']['normalized'].shape).astype('f')
            m['x']['normalized'] = m['x']['normalized'] + noise
        return m


class AdditivePoissonNoise:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, m):
        if random.random() > 0.5:
            std_val = RAND_STATE.uniform(0, 1)
            noise = RAND_STATE.poisson(std_val, size=m['x']['normalized'].shape).astype('f')
            m['x']['normalized'] = m['x']['normalized'] + noise
        return m















#
# class Repr:
#     """Evaluable string representation of an object"""
#
#     def __repr__(self): return f'{self.__class__.__name__}: {self.__dict__}'
#
#
# class FunctionWrapperSingle(Repr):
#     """A function wrapper that returns a partial for input only."""
#
#     def __init__(self, function: Callable, *args, **kwargs):
#         from functools import partial
#         self.function = partial(function, *args, **kwargs)
#
#     def __call__(self, inp: np.ndarray): return self.function(inp)
#
#
# class FunctionWrapperDouble(Repr):
#     """A function wrapper that returns a partial for an input-target pair."""
#
#     def __init__(self, function: Callable, input: bool = True, target: bool = False, *args, **kwargs):
#         from functools import partial
#         self.function = partial(function, *args, **kwargs)
#         self.input = input
#         self.target = target
#
#     def __call__(self, inp: np.ndarray, tar: dict):
#         if self.input: inp = self.function(inp)
#         if self.target: tar = self.function(tar)
#         return inp, tar
#
#
# class Compose:
#     """Baseclass - composes several transforms together."""
#
#     def __init__(self, transforms: List[Callable]):
#         self.transforms = transforms
#
#     def __repr__(self): return str([transform for transform in self.transforms])
#
#
# class ComposeDouble(Compose):
#     """Composes transforms for input-target pairs."""
#
#     def __call__(self, inp: np.ndarray, target: dict):
#         for t in self.transforms:
#             inp, target = t(inp, target)
#         return inp, target
#
#
# class ComposeSingle(Compose):
#     """Composes transforms for input only."""
#
#     def __call__(self, inp: np.ndarray):
#         for t in self.transforms:
#             inp = t(inp)
#         return inp



# def StdLabelToBoundary(inp: np.ndarray):
#     """Normalize the data to a certain range. Default: [0-255]"""
#     results = []
#     inp_out = find_boundaries(inp, connectivity=2, mode='thick')
#     results.append(inp_out)
#     results.append(inp)
#     return np.stack(results, axis=0)
#
#
# def apply_gaussian_blur(inp: np.ndarray):
#     """Normalize the data to a certain range. Default: [0-255]"""
#     results = []
#     blurred = gaussian(inp[0, :], sigma=2, truncate=4, multichannel=False)
#     results.append(blurred)
#     results.append(inp[0, :])
#     results.append(inp[1, :])
#     return np.stack(results, axis=0)
#


# m[0] / np.linalg.norm(m[0]) # normalize(m)  # m /
# np.linalg.norm(m)





# def _recover_ignore_index(input, orig, ignore_index):
#     if ignore_index is not None:
#         mask = orig == ignore_index
#         input[mask] = ignore_index
#
#     return input
#

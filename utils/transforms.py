from collections import Callable
from typing import List

from skimage.segmentation import find_boundaries
from skimage.filters import gaussian
from sklearn.preprocessing import normalize
import numpy as np


class GaussianBlur:
    def __init__(self, sigma=1, truncate=3, multichannel=False, ground_label=False):
        self.sigma = sigma
        self.truncate = truncate
        self.multichannel = multichannel
        self.ground_label = ground_label

    def __call__(self, m):
        if len(m.shape) < 4:
            m = np.expand_dims(m, axis=0)

        piled_results = []

        blurred = gaussian(m[0], sigma=self.sigma, truncate=self.truncate, multichannel=self.multichannel)

        piled_results.append(blurred)
        if self.ground_label:
            for index in range(m.shape[0]):
                piled_results.append((m[index]))

        return np.stack(piled_results, axis=0)


class GroundTruthToBoundary:
    def __init__(self, connectivity=2, mode='thick', ground_label=False):
        self.connectivity = connectivity
        self.mode = mode
        self.ground_label = ground_label

    def __call__(self, m):
        if len(m.shape) < 4:
            m = np.expand_dims(m, axis=0)

        piled_results = []

        boundaries = find_boundaries(m[0], connectivity=2, mode='thick')

        piled_results.append(boundaries)
        if self.ground_label:
            for index in range(m.shape[0]):
                piled_results.append((m[index]))

        return np.stack(piled_results, axis=0)


class OutputClassLabelling:
    def __init__(self, ground_label=False):
        self.ground_label = ground_label

    def __call__(self, m):
        if len(m.shape) < 4:
            m = np.expand_dims(m, axis=0)

        piled_results = []

        boundary_label = np.where(m[0] > m[0].mean(), 1, 0)
        background_label = np.where(boundary_label > 0, 0, 1)

        piled_results.append(boundary_label)
        piled_results.append(background_label)
        if self.ground_label:
            for index in range(m.shape[0]):
                piled_results.append((m[index]))

        return np.stack(piled_results, axis=0)


class Normalize:
    def __init__(self, ground_label=True):
        self.ground_label = ground_label

    def __call__(self, m):
        if len(m.shape) < 4:
            m = np.expand_dims(m, axis=0)

        piled_results = []
        boundaries = m[0]
        if np.max(m[0]) != 0:
            boundaries = (m[0] - np.min(m[0])) / np.ptp(m[0]) # m[0] / np.linalg.norm(m[0]) # normalize(m)  # m / np.linalg.norm(m)

        piled_results.append(boundaries)
        if self.ground_label:
            for index in range(m.shape[0]):
                piled_results.append((m[index]))

        return np.stack(piled_results, axis=0)











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











# def _recover_ignore_index(input, orig, ignore_index):
#     if ignore_index is not None:
#         mask = orig == ignore_index
#         input[mask] = ignore_index
#
#     return input
#

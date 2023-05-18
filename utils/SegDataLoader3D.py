import random
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils import data


class SegDataLoader3D(data.Dataset):

    def __init__(self, files_path, input_transforms=None, target_transforms=None, lim_dataset=None,
                 data_augmentation=None, generate_patches=False):
        super().__init__()
        self.info_data = []
        self.raw_data = []
        self.label_data = []
        self.input_transforms = input_transforms
        self.target_transforms = target_transforms
        self.lim_dataset = lim_dataset
        self.data_augmentation = data_augmentation
        self.generate_patches = generate_patches
        self.raw = 'raw'
        self.label = 'label'

        path_to_files = Path(files_path)
        files = sorted(path_to_files.glob('*.h5'))

        if len(files) <= 0:
            raise RuntimeError("No H5 files found in the directory")

        for file in files:
            self._store_info_data_patches(str(file.resolve()))

    def __len__(self):
        if self.lim_dataset:
            return self.lim_dataset
        return len(self._get_info_data(self.raw))

    def __getitem__(self, index):
        x, y = self._get_data(index)

        if self.input_transforms:
            x = self.input_transforms(x)
        else:
            x = torch.from_numpy(x).float()

        if self.target_transforms:
            y = self.target_transforms(y)
        else:
            y = torch.from_numpy(y).float()

        if self.data_augmentation:
            m = self.data_augmentation({'x': x, 'y': y})
            x = m['x']
            y = m['y']

        return x, y

    def _get_info_data(self, entry_name):
        info_data_type = [entries for entries in self.info_data if entries['entry_name'] == entry_name]
        return info_data_type

    def _store_info_data_patches(self, data_file_path):
        with h5py.File(data_file_path, 'r') as data_file:
            for entry_name, entry_data in data_file.items():
                if entry_name == self.raw:
                    index = 0
                    patch_size = [70, 140, 140]
                    stride_shape = [48, 96, 96]
                    z_slice = entry_data.shape[2]
                    x_slice = entry_data.shape[0]
                    y_slice = entry_data.shape[1]

                    for z_num_stride in range(z_slice // stride_shape[0]):
                        for y_num_stride in range(y_slice // stride_shape[2]):
                            for x_num_stride in range((x_slice // stride_shape[1])):
                                patch_z = self.patch_calc(z_slice, z_num_stride, stride_shape[0], patch_size[0])
                                patch_x = self.patch_calc(x_slice, x_num_stride, stride_shape[1], patch_size[1])
                                patch_y = self.patch_calc(y_slice, y_num_stride, stride_shape[2], patch_size[2])
                                self.info_data.append({'current_file_path': data_file_path, 'entry_name': entry_name,
                                                       'shape': entry_data.shape, 'idx': index,
                                                       'patch_z': patch_z,
                                                       'patch_x': patch_x, 'patch_y': patch_y})
                                index += 1

    def patch_calc(self, dim_size, num_stride, stride, patch_size):
        start = stride * num_stride
        end = start + patch_size
        if end < dim_size:
            patch = (start, end)
        else:
            patch = (dim_size - patch_size, dim_size)
        return patch

    def _sample_open_file(self, data_file_path):
        data_file = h5py.File(data_file_path, 'r')
        for data_file in self.raw_data:
            for entry_name, entry_data in data_file.items():
                if entry_name == self.raw:
                    for index in range(entry_data.shape[2]):
                        self.info_data.append({'current_file_path': data_file_path, 'entry_name': entry_name,
                                               'shape': entry_data.shape, 'idx': index})

        self.raw_data.append({'current_file_path': data_file_path, 'file': data_file})

    def _load_data_h5_patch(self, file_path, index, patch_z, patch_x, patch_y):
        with h5py.File(file_path, 'r') as data_file:
            raw_img = np.array(data_file[self.raw][patch_x[0]:patch_x[1], patch_y[0]:patch_y[1], patch_z[0]:patch_z[1]],
                               dtype=float)
            label_img = np.array(data_file[self.label][patch_x[0]:patch_x[1], patch_y[0]:patch_y[1], patch_z[0]:patch_z[1]],
                                 dtype=int)

            raw_img = raw_img.transpose((2, 0, 1))
            raw_img = np.expand_dims(raw_img, axis=0)

            label_img = label_img.transpose((2, 0, 1))
            label_img = np.expand_dims(label_img, axis=0)

            return raw_img, label_img

    def _get_data(self, index):
        entry = self._get_info_data(self.raw)[index]
        file_path = entry['current_file_path']
        index = entry['idx']
        patch_x = entry['patch_x']
        patch_y = entry['patch_y']
        patch_z = entry['patch_z']
        return self._load_data_h5_patch(file_path, index, patch_z, patch_x, patch_y)

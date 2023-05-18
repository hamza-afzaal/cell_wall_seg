import random
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils import data


class SegmentationDataLoaderH5(data.Dataset):

    def __init__(self, files_path, input_transforms=None, target_transforms=None, lim_dataset=None,
                 data_augmentation=None, generate_patches=False, raw_entry='raw', label_entry='label'):
        super().__init__()
        self.info_data = []
        self.raw_data = []
        self.label_data = []
        self.input_transforms = input_transforms
        self.target_transforms = target_transforms
        self.lim_dataset = lim_dataset
        self.data_augmentation = data_augmentation
        self.generate_patches = generate_patches
        self.raw = 'channel1'
        self.label = 'channel0'

        path_to_files = Path(files_path)
        files = sorted(path_to_files.glob('*.h5'))

        if len(files) <= 0:
            raise RuntimeError("No H5 files found in the directory")

        for file in files:
            if self.generate_patches:
                self._store_info_data_patches(str(file.resolve()))
            else:
                self._store_info_data(str(file.resolve()))

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

    def _store_info_data(self, data_file_path):
        with h5py.File(data_file_path, 'r') as data_file:
            for entry_name, entry_data in data_file.items():
                if entry_name == self.raw:
                    for index in range(entry_data.shape[2]):
                        self.info_data.append({'current_file_path': data_file_path, 'entry_name': entry_name,
                                               'shape': entry_data.shape, 'idx': index})

    def _store_info_data_patches(self, data_file_path):
        with h5py.File(data_file_path, 'r') as data_file:
            for entry_name, entry_data in data_file.items():
                if entry_name == self.raw:
                    for index in range(entry_data.shape[2]):
                        height_center = int(entry_data.shape[0] // 1.2)
                        width_center = int(entry_data.shape[1] // 1.2)
                        for case in range(4):
                            self.info_data.append({'current_file_path': data_file_path, 'entry_name': entry_name,
                                                   'shape': entry_data.shape, 'idx': index,
                                                   'height_center': height_center,
                                                   'width_center': width_center, 'case': case})

    def _sample_open_file(self, data_file_path):
        data_file = h5py.File(data_file_path, 'r')
        for data_file in self.raw_data:
            for entry_name, entry_data in data_file.items():
                if entry_name == self.raw:
                    for index in range(entry_data.shape[2]):
                        self.info_data.append({'current_file_path': data_file_path, 'entry_name': entry_name,
                                               'shape': entry_data.shape, 'idx': index})

        self.raw_data.append({'current_file_path': data_file_path, 'file': data_file})

    def _load_data_h5(self, file_path, index):
        with h5py.File(file_path, 'r') as data_file:
            raw_img = np.array(data_file[self.raw][:, :, index], dtype=float)
            label_img = np.array(data_file[self.label][:, :, index], dtype=int)

            raw_img = np.expand_dims(raw_img, axis=2)
            raw_img = raw_img.transpose((2, 0, 1))

            label_img = np.expand_dims(label_img, axis=2)
            label_img = label_img.transpose((2, 0, 1))

            return raw_img, label_img

    def _load_data_h5_patches(self, file_path, index, height_center, width_center, case):
        with h5py.File(file_path, 'r') as data_file:
            if case == 0:
                raw_img = np.array(data_file[self.raw][:height_center, :width_center, index], dtype=float)
                label_img = np.array(data_file[self.label][:height_center, :width_center, index], dtype=int)
            elif case == 1:
                raw_img = np.array(data_file[self.raw][height_center:, :width_center, index], dtype=float)
                label_img = np.array(data_file[self.label][height_center:, :width_center, index], dtype=int)
            elif case == 2:
                raw_img = np.array(data_file[self.raw][height_center:, width_center:, index], dtype=float)
                label_img = np.array(data_file[self.label][height_center:, width_center:, index], dtype=int)
            elif case == 3:
                raw_img = np.array(data_file[self.raw][:height_center, width_center:, index], dtype=float)
                label_img = np.array(data_file[self.label][:height_center, width_center:, index], dtype=int)
            else:
                raw_img = np.array(data_file[self.raw][:, :, index], dtype=float)
                label_img = np.array(data_file[self.label][:, :, index], dtype=int)

            raw_img = np.expand_dims(raw_img, axis=2)
            raw_img = raw_img.transpose((2, 0, 1))

            label_img = np.expand_dims(label_img, axis=2)
            label_img = label_img.transpose((2, 0, 1))

            return raw_img, label_img

    def _load_direct_data_h5(self, file_path, index):
        data_file = [entries for entries in self.info_data if entries['current_file_path'] == file_path][0]
        raw_img = np.array(data_file[self.raw][:, :, index], dtype=float)
        label_img = np.array(data_file[self.label][:, :, index], dtype=float)

        raw_img = np.expand_dims(raw_img, axis=2)
        raw_img = raw_img.transpose((2, 0, 1))

        label_img = np.expand_dims(label_img, axis=2)
        label_img = label_img.transpose((2, 0, 1))

        return raw_img, label_img

    def _read_data(self, raw_img, label_img):
        raw_img = np.expand_dims(raw_img, axis=2)
        raw_img = raw_img.transpose((2, 0, 1))

        label_img = np.expand_dims(label_img, axis=2)
        label_img = label_img.transpose((2, 0, 1))

        return raw_img, label_img

    def _get_data(self, index):
        entry = self._get_info_data(self.raw)[index]
        file_path = entry['current_file_path']
        index = entry['idx']
        if self.generate_patches:
            height_center = entry['height_center']
            width_center = entry['width_center']
            case = entry['case']
            return self._load_data_h5_patches(file_path, index, height_center, width_center, case)
        else:
            return self._load_data_h5(file_path, index)

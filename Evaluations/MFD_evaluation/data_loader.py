import os

import nibabel
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data


class BRATSVolumes(torch.utils.data.Dataset):
    def __init__(self, directory, test_flag=False, normalize=None, mode='train', img_size=256, modality='t1n'):
        super().__init__()
        self.mode = mode
        self.directory = os.path.expanduser(directory)
        self.normalize = normalize if normalize is not None else (lambda x: x)
        self.test_flag = test_flag
        self.img_size = img_size
        self.modality = modality

        if test_flag:
            self.seqtypes = ['t1n', 't1c', 't2w', 't2f']
        else:
            self.seqtypes = ['t1n', 't1c', 't2w', 't2f', 'seg']
        self.seqtypes_set = set(self.seqtypes)

        if self.modality not in self.seqtypes_set:
            raise ValueError(
                "Unsupported modality '{}'. Available options are: {}".format(
                    self.modality, ', '.join(sorted(self.seqtypes_set))
                )
            )

        self.database = self._build_database()

    def _build_database(self):
        if self.mode == 'fake':
            return self._collect_fake_database()
        return self._collect_real_database()

    def _collect_real_database(self):
        database = []
        for root, dirs, files in os.walk(self.directory):
            dirs.sort()
            files.sort()
            if dirs:
                continue

            datapoint = {}
            for filename in files:
                if not self._is_nii_file(filename):
                    continue
                seqtype = self._extract_real_seqtype(filename)
                datapoint[seqtype] = os.path.join(root, filename)

            if self.modality in datapoint:
                database.append({self.modality: datapoint[self.modality]})
        return database

    def _collect_fake_database(self):
        alias_names = self._get_fake_modality_aliases()
        directory_name = os.path.basename(os.path.normpath(self.directory)).lower()

        if directory_name in alias_names:
            return self._collect_fake_from_single_modality_root()

        return self._collect_fake_from_common_root(alias_names)

    def _collect_fake_from_common_root(self, alias_names):
        database = []
        if not os.path.isdir(self.directory):
            return database

        for subject_name in sorted(os.listdir(self.directory)):
            subject_dir = os.path.join(self.directory, subject_name)
            if not os.path.isdir(subject_dir):
                continue

            selected_path = None
            for alias_name in alias_names:
                modality_dir = os.path.join(subject_dir, alias_name)
                if not os.path.isdir(modality_dir):
                    continue

                selected_path = self._select_fake_file_from_directory(modality_dir)
                if selected_path is not None:
                    break

            if selected_path is not None:
                database.append({self.modality: selected_path})
        return database

    def _collect_fake_from_single_modality_root(self):
        database = []
        for root, dirs, files in os.walk(self.directory):
            dirs.sort()
            files.sort()

            if root == self.directory:
                for filename in files:
                    if self._is_nii_file(filename):
                        database.append({self.modality: os.path.join(root, filename)})
                continue

            selected_path = self._select_fake_file_from_filenames(root, files)
            if selected_path is not None:
                database.append({self.modality: selected_path})
        return database

    def _select_fake_file_from_directory(self, directory):
        filenames = sorted(os.listdir(directory))
        return self._select_fake_file_from_filenames(directory, filenames)

    def _select_fake_file_from_filenames(self, root, filenames):
        nii_files = [filename for filename in filenames if self._is_nii_file(filename)]
        if not nii_files:
            return None

        preferred_filenames = ('sample.nii.gz', 'sample.nii')
        for preferred_filename in preferred_filenames:
            if preferred_filename in nii_files:
                return os.path.join(root, preferred_filename)

        if len(nii_files) == 1:
            return os.path.join(root, nii_files[0])

        return os.path.join(root, nii_files[0])

    def _extract_real_seqtype(self, filename):
        basename = os.path.basename(filename)
        return basename.rsplit('-', 1)[-1].split('.')[0]

    def _get_fake_modality_aliases(self):
        alias_map = {
            't1n': ('t1', 't1n'),
            't1c': ('t1c',),
            't2f': ('flair', 't2f'),
            't2w': ('t2w',),
            'seg': ('seg',),
        }
        return alias_map[self.modality]

    def _is_nii_file(self, filename):
        return filename.endswith('.nii') or filename.endswith('.nii.gz')

    def _load_volume(self, filepath):
        nib_img = nibabel.load(filepath)
        return nib_img.get_fdata()

    def _prepare_real_volume(self, volume):
        clipped = np.clip(volume, np.quantile(volume, 0.001), np.quantile(volume, 0.999))
        normalized = (clipped - np.min(clipped)) / (np.max(clipped) - np.min(clipped))
        tensor = torch.tensor(normalized)

        image = torch.zeros(1, 256, 256, 155)
        image[:, 8:-8, 8:-8, :] = tensor
        return image

    def _prepare_fake_volume(self, volume):
        tensor = torch.tensor(volume)
        shape = tuple(tensor.shape)

        if shape == (160, 256, 256):
            image = torch.zeros(1, 256, 256, 155, dtype=torch.float32)
            image[:, :, :, :] = tensor[:155, :, :].permute(1, 2, 0)
            return image
        raise ValueError(
            'Unexpected fake volume shape {} for file in modality {}'.format(shape, self.modality)
        )

    def __getitem__(self, index):
        filedict = self.database[index]
        name = filedict[self.modality]
        volume = self._load_volume(name)

        if self.mode == 'fake':
            image = self._prepare_fake_volume(volume)
        else:
            image = self._prepare_real_volume(volume)

        image = self.normalize(image)

        if self.mode == 'fake':
            return image, name
        return image

    def __len__(self):
        return len(self.database)

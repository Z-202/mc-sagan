import torch
import torch.utils.data
import numpy as np
import os
import os.path
import nibabel

class BRATSVolumes(torch.utils.data.Dataset):
    def __init__(self, directory, mode='train', gen_type=None):
        super().__init__()
        self.mode = mode
        self.directory = os.path.expanduser(directory)
        self.gentype = gen_type
        self.database = []

        for root, dirs, files in os.walk(self.directory):

            if not dirs:
                files.sort()
                datapoint = dict()

                for f in files:
                    seqtype = f.split('-')[4].split('.')[0]
                    datapoint[seqtype] = os.path.join(root, f)
                self.database.append(datapoint)

    def __getitem__(self, x):
        filedict = self.database[x]
        missing = 'none'

        if 't1n' in filedict:
            t1n_np = nibabel.load(filedict['t1n']).get_fdata()
            t1n_np_clipnorm = clip_and_normalize(t1n_np)

            t1n = torch.full((1, 240, 240, 160), -1.0)
            t1n[:, :, :, :155] = torch.tensor(t1n_np_clipnorm)
            t1n_padded = torch.full((1, 256, 256, 160), -1.0)
            t1n_padded[:, 8:-8, 8:-8, :] = t1n
            t1n = t1n_padded
            t1n = t1n.permute(0, 3, 1, 2).contiguous()
        else:
            missing = 't1n'
            t1n = torch.zeros(1)

        if 't1c' in filedict:
            t1c_np = nibabel.load(filedict['t1c']).get_fdata()
            t1c_np_clipnorm = clip_and_normalize(t1c_np)
            t1c = torch.full((1, 240, 240, 160), -1.0)
            t1c[:, :, :, :155] = torch.tensor(t1c_np_clipnorm)
            t1c_padded = torch.full((1, 256, 256, 160), -1.0)
            t1c_padded[:, 8:-8, 8:-8, :] = t1c
            t1c = t1c_padded
            t1c = t1c.permute(0, 3, 1, 2).contiguous()
        else:
            missing = 't1c'
            t1c = torch.zeros(1)

        if 't2w' in filedict:
            t2w_np = nibabel.load(filedict['t2w']).get_fdata()
            t2w_np_clipnorm = clip_and_normalize(t2w_np)
            t2w = torch.full((1, 240, 240, 160), -1.0)
            t2w[:, :, :, :155] = torch.tensor(t2w_np_clipnorm)
            t2w_padded = torch.full((1, 256, 256, 160), -1.0)
            t2w_padded[:, 8:-8, 8:-8, :] = t2w
            t2w = t2w_padded
            t2w = t2w.permute(0, 3, 1, 2).contiguous()
        else:
            missing = 't2w'
            t2w = torch.zeros(1)

        if 't2f' in filedict:
            t2f_np = nibabel.load(filedict['t2f']).get_fdata()
            t2f_np_clipnorm = clip_and_normalize(t2f_np)
            t2f = torch.full((1, 240, 240, 160), -1.0)
            t2f[:, :, :, :155] = torch.tensor(t2f_np_clipnorm)
            t2f_padded = torch.full((1, 256, 256, 160), -1.0)
            t2f_padded[:, 8:-8, 8:-8, :] = t2f
            t2f = t2f_padded
            t2f = t2f.permute(0, 3, 1, 2).contiguous()
        else:
            missing = 't2f'
            t2f = torch.zeros(1)

        if 'seg' in filedict:
            seg_np = nibabel.load(filedict['seg']).get_fdata()
            seg_np_bin = (seg_np > 0).astype(np.float32)
            seg = torch.zeros(1, 240, 240, 160)
            seg[:, :, :, :155] = torch.tensor(seg_np_bin)
            seg_padded = torch.zeros(1, 256, 256, 160)
            seg_padded[:, 8:-8, 8:-8, :] = seg
            seg = seg_padded
            seg = seg.permute(0, 3, 1, 2).contiguous()
        else:
            missing = 'seg'
            seg = torch.zeros(1)

        if self.mode == 'eval' or self.mode == 'auto':
            if 't1n' in filedict:
                subj = filedict['t1n']
            else:
                subj = filedict['t2f']
        else:
            subj = 'dummy_string'

        return {'t1n': t1n.float(),
                't1c': t1c.float(),
                't2w': t2w.float(),
                't2f': t2f.float(),
                'seg': seg.float(),
                'missing': missing,
                'subj': subj,
                'filedict': filedict}

    def __len__(self):
        return len(self.database)

def clip_and_normalize(img):
    img_clipped = np.clip(img, np.quantile(img, 0.001), np.quantile(img, 0.999))
    img_normalized_01 = (img_clipped - np.min(img_clipped)) / (np.max(img_clipped) - np.min(img_clipped))

    img_normalized_m11 = img_normalized_01 * 2.0 - 1.0
    return img_normalized_m11

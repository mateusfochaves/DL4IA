'''Dataset class to load the S2-Agri Pixel-Set data: https://zenodo.org/records/5815488
'''

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import numpy as np

import os
import collections.abc

from myutils.utils import pad_tensor


class Padding:
    ''' Used to build data loaders of irregular time series (potentially of various lengths).
    '''
    def __init__(self, pad_value=0):
        self.pad_value = pad_value
    
    def pad_collate(self, batch: list[tuple[torch.Tensor]]):
        # Utility function to be used as collate_fn for the PyTorch dataloader
        # to handle sequences of varying length.
        # Sequences are padded with zeros by default.
        #
        # Modified default_collate from the official pytorch repo
        # https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py
        elem = batch[0]
        elem_type = type(elem)
        if isinstance(elem, torch.Tensor):
            out = None
            if len(elem.shape) > 0:
                sizes = [e.shape[0] for e in batch]
                m = max(sizes)
                if not all(s == m for s in sizes):
                    # pad tensors which have a temporal dimension
                    batch = [pad_tensor(e, m, self.pad_value) for e in batch]
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            return torch.stack(batch, 0, out=out)
        elif (
            elem_type.__module__ == "numpy"
            and elem_type.__name__ != "str_"
            and elem_type.__name__ != "string_"
        ):
            return torch.as_tensor(batch)

        elif isinstance(elem, collections.abc.Sequence):
            # check to make sure that the elements in batch have consistent size
            it = iter(batch)
            elem_size = len(next(it))
            if not all(len(elem) == elem_size for elem in it):
                raise RuntimeError("each element in list of batch should be of equal size")
            transposed = zip(*batch)
            return [self.pad_collate(samples) for samples in transposed]

        raise TypeError("Format not managed : {}".format(elem_type))


class PixelSetData(Dataset):
    '''TODO: Complete the dataset class to load the S2-Agri Pixel-Set data.
    '''
    def __init__(self, folder, quantification_value=10000, set='train'):
        super(PixelSetData, self).__init__()

        self.folder = folder
        self.set = set

        self.quantification_value = quantification_value # 10000 for Sentinel-2, see the link below for more info:
        # https://gis.stackexchange.com/questions/233874/what-is-the-range-of-values-of-sentinel-2-level-2a-images

        if set == 'train':
            labels = np.load(os.path.join(folder, 'train_labels.npy'))
        elif set == 'test':
            labels = np.load(os.path.join(folder, 'test_labels.npy'))
        else:
            raise NotImplementedError
        
        self.labels = torch.from_numpy(labels)

        self.label_names = [
            'Winter Durum Wheat',
            'Spring Cereal',
            'Summer Cereal',
            'Winter Cereal',
            'Cereal',
            'Leguminous Fodder',
            'Other Fodder',
            'Winter Rapeseed'
        ]


    def __len__(self): #donne la taille du dataset
        return len(self.labels)
        
    
    def __getitem__(self, i): #renvoie les données de l'échantillon i 
        if self.set in ['train', 'test']:
            sample = torch.load(os.path.join(self.folder, 'data', f'sample_{i}.pt'))
            doys = torch.load(os.path.join(self.folder, 'data', f'doy_{i}.pt'))
        else:
            raise NotImplementedError
        
        label = self.labels[i]
        sample = sample / self.quantification_value

        return sample, doys, label
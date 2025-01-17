import random
import os
from pathlib import Path

import h5py
import torch
import torch.utils.data as data

import data_transform
from torch.utils.data import DataLoader


class TrainFromHdf5(data.Dataset):
    """
    Creates a training set from a hdf5 file
    """

    def __init__(
            self, file_path, patch_size,
            num_crops, transform=None, fixed_seed=False, sub_chan=False, crop_train=True):
        """
        Keyword arguments:
        hdf_file -- the location containing the hdf5 file
        patch_size -- the size of the patches to extract for training
        num_crops -- the number of patches to extract for training
        transform -- an optional transform to apply to the data
        """
        super()
        self.file_path = file_path
        with h5py.File(
                file_path, mode='r', libver='latest', swmr=True) as h5_file:
            self.num_samples = h5_file['train'].attrs['lf_shape'][0]
            self.grid_size = h5_file['train'].attrs['lf_shape'][1]
        self.colour = '/train/images'
        self.warped = '/train/warped'
        self.transform = transform
        self.patch_size = patch_size
        self.num_crops = num_crops
        self.sub_chan = sub_chan
        self.crop_train = crop_train
        if fixed_seed:
            random.seed(100)
        else:
            random.seed()

    def __getitem__(self, index):
        """
        Return item at index in 0 to len(self)
        In this case a set of crops from an lf sample
        Return type is a dictionary of depth and colour arrays
        """
        with h5py.File(
                self.file_path, mode='r',
                libver='latest', swmr=True) as h5_file:
            idx = index // self.num_crops
            colour = torch.tensor(
                h5_file[self.colour][idx], dtype=torch.float32)
            warped = torch.tensor(
                h5_file[self.warped][idx], dtype=torch.float32)
            grid_size = self.grid_size
            sample = {
                'colour': colour,
                'warped': warped,
                'grid_size': grid_size}

            if self.crop_train:
                sample = data_transform.get_random_crop(sample, self.patch_size)
            sample = data_transform.normalise_sample(sample)
            sample = data_transform.random_gamma(sample)

            if self.sub_chan:
                sample = data_transform.subsample_channels(
                    sample, 3
                )
            if self.transform:
                sample = self.transform(sample)

            return sample

    def __len__(self):
        """Return the number of samples in the dataset"""
        return self.num_samples * self.num_crops


class ValFromHdf5(data.Dataset):
    """
    Creates a validation set from a hdf5 file
    """

    def __init__(
            self, file_path, patch_size, name,
            transform=None, sub_chan=False, val_transform=False):
        """
        Keyword arguments:
        hdf_file -- the location containing the hdf5 file
        transform -- an optional transform to apply to the data
        """
        super()
        self.file_path = file_path
        with h5py.File(
                file_path, mode='r', libver='latest', swmr=True) as h5_file:
            self.num_samples = h5_file[name].attrs['lf_shape'][0]
            self.grid_size = h5_file[name].attrs['lf_shape'][1]
            self.im_size = h5_file[name].attrs['lf_shape'][-1]
        self.colour = '/{}/images'.format(name)
        self.warped = '/{}/warped'.format(name)
        self.transform = transform
        self.patch_size = patch_size
        self.val_transform = val_transform
        if self.val_transform:
            self.crop_cords = data_transform.create_random_coords(
                self.im_size, self.num_samples, self.patch_size
            )
        self.sub_chan = sub_chan

    def __getitem__(self, index):
        """
        Return item at index in 0 to len(self)
        In this case a set of crops from an lf sample
        Return type is a dictionary of depth and colour arrays
        """
        with h5py.File(
                self.file_path, mode='r',
                libver='latest', swmr=True) as h5_file:
            colour = torch.tensor(
                h5_file[self.colour][index], dtype=torch.float32)
            warped = torch.tensor(
                h5_file[self.warped][index], dtype=torch.float32)
            grid_size = self.grid_size
            sample = {
                'colour': colour,
                'warped': warped,
                'grid_size': grid_size}

        # Running out of GPU memory on validation
        if self.val_transform:
            sample = data_transform.crop(sample, self.crop_cords[index])

        if self.sub_chan:
            sample = data_transform.subsample_channels(
                sample, 3
            )
        
        sample = data_transform.normalise_sample(sample)
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        """Return the number of samples in the dataset"""
        return self.num_samples


def create_dataloaders(args, config):
    """Creates a train and val dataloader from a h5file and a config file"""
    print("Loading dataset")
    file_path = os.path.join(config['PATH']['hdf5_dir'],
                             config['PATH']['hdf5_name'])
    if not Path(file_path).is_file():
        print(file_path, " is not a valid location")
        print("Please enter a valid location of a h5 file through main.ini")
        exit(-1)
    train_set = TrainFromHdf5(
        file_path=file_path,
        patch_size=int(config['NETWORK']['patch_size']),
        num_crops=int(config['NETWORK']['num_crops']),
        transform=data_transform.angular_remap,
        sub_chan=config["NETWORK"]["sub_chan"],
        crop_train=config["NETWORK"]["crop_train"])
    batch_size = {'train': int(config['NETWORK']['train_batch_size'])}
    all_sets = [('train', train_set)]
    val_size = int(config['NETWORK']['val_batch_size'])
    for tup in config["VALSETS"].items():
        name = tup[1]
        new_set = ValFromHdf5(
            file_path=file_path, name=name,
            patch_size=int(config["NETWORK"]["val_patch_size"]),
            transform=data_transform.angular_remap,
            sub_chan=config["NETWORK"]["sub_chan"])
        batch_size[name] = val_size
        all_sets.append((name, new_set))

    data_loaders = {}
    threads = int(config['NETWORK']['num_workers'])
    for name, dset in all_sets:
        data_loaders[name] = DataLoader(
            dataset=dset, num_workers=threads,
            batch_size=batch_size[name],
            shuffle=True)

    return data_loaders

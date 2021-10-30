import torch
import os
from dataset import Multimodal_Datasets
from torch.utils.data import DataLoader
from parameters import param


def get_data(dataset, split='train'):
    alignment = 'a' if param["aligned"] else 'na'
    data_path = os.path.join(param["data_path"], dataset) + f'_{split}_{alignment}.dt'
    if os.path.exists(data_path):
        print(f"  - Found stored {split} data")
        data = torch.load(data_path)
    else:
        print(f"  - Creating new {split} data")
        data = Multimodal_Datasets(param["data_path"], dataset, split, param["aligned"])
        torch.save(data, data_path)

    return data


def get_dataloaders(dataset):
    train_data = get_data(dataset, 'train')
    valid_data = get_data(dataset, 'valid')
    test_data = get_data(dataset, 'test')

    train_loader = DataLoader(train_data, batch_size=param["batch_size"], shuffle=False)
    valid_loader = DataLoader(valid_data, batch_size=param["batch_size"], shuffle=False)
    test_loader = DataLoader(test_data, batch_size=param["batch_size"], shuffle=False)

    return train_loader, valid_loader, test_loader


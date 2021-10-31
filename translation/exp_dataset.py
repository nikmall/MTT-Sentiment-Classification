import torch
import os
from dataset import Multimodal_Datasets
from torch.utils.data import DataLoader

param = {
    "dataset" : 'mosei_senti', # 'mosei',
    "aligned" : 'True',
    "data_path" : '..//data',
    "batch_size" : 32 }


def get_data(dataset, split='train'):
    alignment = 'a' if param["aligned"] else 'na'
    data_path = os.path.join(param["data_path"], dataset) + f'_{split}_{alignment}.dt'
    if not os.path.exists(data_path):
        print(f"  - Creating new {split} data")
        data = Multimodal_Datasets(param["data_path"], dataset, split, param["aligned"])
        torch.save(data, data_path)
    else:
        print(f"  - Found cached {split} data")
        data = torch.load(data_path)
    return data


def get_dataloaders():
    train_data = get_data( param["dataset"], 'train')
    valid_data = get_data( param["dataset"], 'valid')
    test_data = get_data( param["dataset"], 'test')


    train_loader = DataLoader(train_data, batch_size=param["batch_size"], shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=param["batch_size"], shuffle=True)
    test_loader = DataLoader(test_data, batch_size=param["batch_size"], shuffle=True)

    return train_loader, valid_loader, test_loader

"""
for i_batch, (batch_X, batch_Y, batch_META) in enumerate(train_loader):
    sample_ind, text, audio, vision = batch_X
    text, audio, vision = text.permute(1,0,2), audio.permute(1,0,2), vision.permute(1,0,2)
    print("")
"""
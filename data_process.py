import random
import numpy as np
import torch
import os
from dataset import Multimodal_Datasets
from torch.utils.data import DataLoader
from parameters import param

from sklearn.preprocessing import StandardScaler, MinMaxScaler

def seed_worker(worker_seed):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(worker_seed)
    torch.cuda.manual_seed_all(worker_seed)

    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_data(dataset, split='train'):
    alignment = 'a' if param["aligned"] else 'na'
    data_path = os.path.join(param["data_path"], dataset) + f'_{split}_{alignment}.dt'
    if os.path.exists(data_path):
        print(f"  - Found stored {split} data")
        data = torch.load(data_path, encoding='ascii')
    else:
        print(f"  - Creating new {split} data")
        data = Multimodal_Datasets(param["data_path"], dataset, split, param["aligned"])
        torch.save(data, data_path)

    return data


def get_dataloaders(dataset, seed_custom, scale=True):
    # g = torch.Generator()
    # g.manual_seed(seed_custom)

    train_data = get_data(dataset, 'train')
    valid_data = get_data(dataset, 'valid')
    test_data = get_data(dataset, 'test')

    if scale:
        scale_data(train_data, valid_data, test_data)

    train_loader = DataLoader(train_data, batch_size=param["batch_size"], worker_init_fn=seed_worker,
                              shuffle=True) # , generator=g
    valid_loader = DataLoader(valid_data, batch_size=param["batch_size"], worker_init_fn=seed_worker,
                              shuffle=True)
    test_loader = DataLoader(test_data, batch_size=param["batch_size"], worker_init_fn=seed_worker,
                             shuffle=True)

    return train_loader, valid_loader, test_loader

def scale_data(train_data, valid_data, test_data, type='stand'):
    # audio
    train_audio = train_data.audio.detach().numpy()
    audio_shape = train_audio.shape
    train_audio = train_audio.reshape(audio_shape[0] * audio_shape[1], audio_shape[2])
    if type == 'min_max':
        audio_scaler = MinMaxScaler()
    else:
        audio_scaler = StandardScaler()
    audio_scaler.fit(train_audio)
    train_audio = audio_scaler.transform(train_audio)
    train_data.audio = torch.tensor(train_audio.reshape(audio_shape[0], audio_shape[1], audio_shape[2])).cpu().detach()

    valid_audio = valid_data.audio.detach().numpy()
    valid_audio = valid_audio.reshape(-1, audio_shape[2])
    valid_audio = audio_scaler.transform(valid_audio)
    valid_data.audio = torch.tensor(valid_audio.reshape(-1, audio_shape[1], audio_shape[2])).cpu().detach()

    test_audio = test_data.audio.detach().numpy()
    test_audio = test_audio.reshape(-1, audio_shape[2])
    test_audio = audio_scaler.transform(test_audio)
    test_data.audio = torch.tensor(test_audio.reshape(-1, audio_shape[1], audio_shape[2])).cpu().detach()

    # Text
    train_text = train_data.text.detach().numpy()
    text_shape = train_text.shape
    train_text = train_text.reshape(text_shape[0] * text_shape[1], text_shape[2])
    if type == 'min_max':
        text_scaler = MinMaxScaler()
    else:
        text_scaler = StandardScaler()
    text_scaler.fit(train_text)
    train_text = text_scaler.transform(train_text)
    train_data.text = torch.tensor(train_text.reshape(-1, text_shape[1], text_shape[2])).cpu().detach()

    valid_text = valid_data.text.detach().numpy()
    valid_text = valid_text.reshape(-1, text_shape[2])
    valid_text = text_scaler.transform(valid_text)
    valid_data.text = torch.tensor(valid_text.reshape(-1, text_shape[1], text_shape[2])).cpu().detach()

    test_text = test_data.text.detach().numpy()
    test_text = test_text.reshape(-1, text_shape[2])
    test_text = text_scaler.transform(test_text)
    test_data.text = torch.tensor(test_text.reshape(-1, text_shape[1], text_shape[2])).cpu().detach()

    # Vision
    train_vision = train_data.vision.detach().numpy()
    vision_shape = train_vision.shape
    train_vision = train_vision.reshape(vision_shape[0] * vision_shape[1], vision_shape[2])
    if type == 'min_max':
        vision_scaler = MinMaxScaler()
    else:
        vision_scaler = StandardScaler()
    train_vision = vision_scaler.fit_transform(train_vision)
    train_data.vision = torch.tensor(train_vision.reshape(-1, vision_shape[1], vision_shape[2])).cpu().detach()

    valid_vision = valid_data.vision.detach().numpy()
    valid_vision = valid_vision.reshape(-1, vision_shape[2])
    valid_vision = vision_scaler.transform(valid_vision)
    valid_data.vision = torch.tensor(valid_vision.reshape(-1, vision_shape[1], vision_shape[2])).cpu().detach()

    test_vision = test_data.vision.detach().numpy()
    test_vision = test_vision.reshape(-1, vision_shape[2])
    test_vision = vision_scaler.transform(test_vision)
    test_data.vision = torch.tensor(test_vision.reshape(-1, vision_shape[1], vision_shape[2])).cpu().detach()

    # return

def scale_2d_standard(x, mu=None, std=None):
    if mu is None and std is None:
        mu = x.mean(axis=(0, 1), keepdims=True)
        std = x.std(axis=(0, 1), keepdims=True)
    x = (x - mu) / (std + 1e-10)

    return x, mu, std

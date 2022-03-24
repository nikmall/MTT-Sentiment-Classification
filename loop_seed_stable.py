import random
import os.path
import joblib
import logging
import sys
import optuna
import argparse
import numpy as np
import random

import torch

import tools
from data_process import get_dataloaders
from main import process
from mtt_fuse.mtt_fuse import start_mtt_fuse


def processing():
    dataset = 'mosei_senti'

    epochs = 50
    model_type = 'mtt_fuse'

    param_mtt_fuse = {
        "enc_emb_dim": 300,
        "dec_emb_dim": 300,
        "hid_dim": 300,
        "enc_layers": 3,
        "dec_layers": 3,
        "enc_heads": 6,
        "dec_heads": 6,
        "enc_pf_dim": 490,
        "dec_pf_dim": 490,
        "enc_dropout": 0.31,
        "dec_dropout": 0.31,
        "att_dropout": 0.31,

        "sent_hid_dim": 156,
        "sent_final_hid": 132,
        "sent_dropout": 0.31,
        "sent_n_layers": 2,
        "bidirect": True,

        "transformer_regression": False,

        "n_epochs": 150,
        'lr_patience': 20,
        'loss_dec_weight': 0.15,
        'loss_dec_cycle_weight': 0.10,
        'loss_regress_weight': 0.9,

        'fuse_modalities': True,
        "cyclic": True
    }

    seed = 62
    tools.seed_all(seed)
    if torch.cuda.is_available():
        print("using cuda")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f'Processing dataset {dataset} for training on {model_type} model type')
    seed_scores = np.empty(shape=(0, 2))
    train_loader, valid_loader, test_loader = get_dataloaders(dataset, seed, scale=True)

    for x in range(3):
        f1_score = start_mtt_fuse(train_loader, valid_loader, test_loader, param_mtt_fuse, device, epochs)
        seed_scores = np.vstack((seed_scores, np.array([(seed, f1_score)])))


        print(f"For seed {seed} the f1_score is {f1_score}")
        # scores_np = np.array(list(seed_scores))
        np.savetxt("loop_scores.txt", seed_scores, delimiter=",")



if __name__ == '__main__':
    processing()
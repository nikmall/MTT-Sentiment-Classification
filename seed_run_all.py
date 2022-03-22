import random
import os.path
import joblib
import logging
import sys
import optuna
import argparse
import numpy as np
import random

import tools
from main import process

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

    seeds = [62, 62, 62, 62]
    max_score = 0
    seed_scores = np.empty(shape=(0, 2))
    best_seed = 0
    for seed in seeds:
        tools.seed_all()
        f1_score = process(epochs, dataset, model_type, param_mtt_fuse, seed)
        seed_scores = np.vstack((seed_scores, np.array([(seed, f1_score)])))

        if f1_score > max_score:
            max_score = f1_score
            best_seed = seed

        print(f"For seed {seed} the f1_score is {f1_score}")
        # scores_np = np.array(list(seed_scores))
        np.savetxt("seed_scores.txt", seed_scores, delimiter=",")

    print(f"The max score is {max_score} for seed {best_seed}")


if __name__ == '__main__':
    processing()


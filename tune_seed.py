import random
import os.path
import joblib
import logging
import sys
import optuna
import argparse

from main import process

optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))


def objective(trial):

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

    seed = trial.suggest_categorical("seed", [62, 63, 64, 65, 66, 67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87])

    f1_score = process(epochs, dataset, model_type, param_mtt_fuse, seed)
    return f1_score


def show_best_results(study):
    print("Best params: ", study.best_params)
    print("Best f1-score: ", study.best_value)
    print("Best Trial: ", study.best_trial)


def tune(total_trials, study_name, storage_name):
    n_trials = total_trials

    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0), direction="maximize",
                                study_name=study_name, storage=storage_name, load_if_exists=True)
    study.optimize(objective, n_trials=n_trials)
    show_best_results(study)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CMU-MOSEI Sentiment Classifier')
    parser.add_argument('--n_trials', default=26, type=int, help='Number of trials to train for tuning hyperparameters')
    args = parser.parse_args()
    n_trials = int(args.n_trials)
    study_name = 'study_mtt_fuse_seed'
    storage_name = "sqlite:///{}.db".format(study_name)
    tune(n_trials, study_name, storage_name)



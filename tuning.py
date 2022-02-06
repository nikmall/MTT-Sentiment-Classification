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

    epochs = 2
    model_type = 'mtt_fuse'

    param_mtt_fuse = {
        "enc_emb_dim": 300,
        "dec_emb_dim": 300,
        "hid_dim": 300,
        "enc_layers": trial.suggest_int("enc_layers", 2, 4),
        "dec_layers": trial.suggest_int("dec_layers", 2, 4),
        "enc_heads": trial.suggest_int("enc_heads", 2, 6),
        "dec_heads": trial.suggest_int("dec_heads", 2, 6),
        "enc_pf_dim": trial.suggest_int("enc_pf_dim", 310, 600),
        "dec_pf_dim": trial.suggest_int("dec_pf_dim", 310, 600),
        "enc_dropout": trial.suggest_float("enc_dropout", 0.15, 0.40, step=0.01),
        "dec_dropout": trial.suggest_float("dec_dropout", 0.15, 0.40, step=0.01),
        "att_dropout": trial.suggest_float("att_dropout", 0.15, 0.40, step=0.01),

        "sent_hid_dim": trial.suggest_int("sent_hid_dim", 90, 204),
        "sent_final_hid": trial.suggest_int("sent_final_hid", 80, 170),
        "sent_dropout": trial.suggest_float("sent_dropout", 0.15, 0.40, step=0.01),
        "sent_n_layers": trial.suggest_int("sent_n_layers", 2, 3),
        "bidirect": trial.suggest_categorical("bidirect", [True, False]),

        "transformer_regression": False,

        "n_epochs": 150,
        'lr_patience': 20,
        'loss_dec_weight': trial.suggest_float("loss_dec_weight", 0.09, 0.18, step=0.01),
        'loss_dec_cycle_weight': trial.suggest_float("loss_dec_cycle_weight", 0.07, 0.14, step=0.01),
        'loss_regress_weight': 0.9,

        'fuse_modalities': True,
        "cyclic": trial.suggest_categorical("cyclic", [True, False])
    }

    f1_score = process(epochs, dataset, model_type, param_mtt_fuse)
    # f1_score = random.random()
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
    parser.add_argument('--n_trials', default=1000, type=int, help='Number of trials to train for tuning hyperparameters')
    args = parser.parse_args()
    n_trials = int(args.n_trials)
    study_name = 'study_mtt_fuse'
    storage_name = "sqlite:///{}.db".format(study_name)
    tune(n_trials, study_name, storage_name)



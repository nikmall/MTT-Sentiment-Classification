import argparse
import torch
import random
import numpy as np

import tools
from data_process import get_dataloaders
from parameters import param_mctn, param_mtt, param_mtt_fuse
from mctn_rnn.mctn import start_mctn
from mtt.mtt_cyclic import start_mtt_cyclic
from mtt_fuse.mtt_fuse import start_mtt_fuse



""" 
seed = 88
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
"""

if torch.cuda.is_available():
    print("using cuda")
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def main():
    seed = 88

    parser = argparse.ArgumentParser(description='CMU-MOSEI Sentiment Classifier')

    parser.add_argument('--model', type=str, default='mtt_fuse',
                        help='Options are: mctn, mtt_cyclic, mtt_fuse')

    parser.add_argument('--dataset', type=str, default='mosei', help='Enter either mosei or mosi')

    parser.add_argument('--epochs', default=50, type=int, help='Number of epochs to train. If none, use from param file')
    parser.add_argument('--cont_loaded', type=bool, default=False, help='To load existing saved model and continue')

    args = parser.parse_args()

    model_type = str.lower(args.model.strip())

    epochs = int(args.epochs)

    dataset = str.lower(args.dataset.strip())
    if dataset == 'mosei':
        dataset = 'mosei_senti'

    if model_type == 'mtt_cyclic':
        params = param_mtt
    elif model_type == 'mtt_fuse':
        params = param_mtt_fuse
    elif model_type == 'mctn':
        params = param_mctn

    score = process(epochs, dataset, model_type, params, seed)


def process(epochs, dataset, model_type, params, seed):

    tools.seed_all(seed)

    print(f'Processing dataset {dataset} for training on {model_type} model type')

    train_loader, valid_loader, test_loader = get_dataloaders(dataset, seed, scale=True)
    print("Loaded the Dataloaders")

    if model_type == 'mtt_cyclic':
        score = start_mtt_cyclic(train_loader, valid_loader, test_loader, params, device, epochs)
    elif model_type == 'mtt_fuse':
        score = start_mtt_fuse(train_loader, valid_loader, test_loader, params, device, epochs)
    elif model_type == 'mctn':
        score = start_mctn(train_loader, valid_loader, test_loader, params, device, epochs)

    return score

if __name__ == '__main__':
    main()

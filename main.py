import argparse
import torch
import random
import numpy as np

from data_process import get_dataloaders
from parameters import param_mctn, param_mtt, param_mtt_fuse
from mctn_rnn.mctn import start_mctn
from mtt.mtt_cyclic import start_mtt_cyclic
from mtt_fuse.mtt_fuse import start_mtt_fuse

seed = 11
torch.cuda.manual_seed(seed)
random.seed(seed)
np.random.seed(0)
device = torch.device("cpu")
if torch.cuda.is_available():
    print("using cuda")
    device = torch.device("cuda")

torch.backends.cudnn.enabled = False

def main():
    parser = argparse.ArgumentParser(description='CMU-MOSEI Sentiment Classifier')

    parser.add_argument('--model', type=str, default='mtt_cyclic',
                        help='Options are: mctn, mtt_cyclic, mtt_fuse')


    parser.add_argument('--dataset', type=str, default='mosei', help='Enter either mosei or mosi')

    parser.add_argument('--epochs', type=int, help='Number of epochs to train. If none, use from param file')
    parser.add_argument('--cont_loaded', type=bool, default=False, help='To load existing saved model and continue')
    args = parser.parse_args()

    model_type = str.lower(args.model.strip())

    epochs = int(args.epochs)
    cont_loaded = args.cont_loaded

    dataset = str.lower(args.dataset.strip())
    if dataset == 'mosei':
        dataset = 'mosei_senti'



    print(f'Processing dataset {dataset} for training on {model_type} model type')

    train_loader, valid_loader, test_loader = get_dataloaders(dataset, seed, True)
    print("Loaded the Dataloaders")


    if model_type == 'mtt_cyclic':
        start_mtt_cyclic(train_loader, valid_loader, test_loader, param_mtt, device, epochs)
    elif model_type == 'mtt_fuse':
        print()
        start_mtt_fuse(train_loader, valid_loader, test_loader, param_mtt_fuse, device, epochs)
    elif model_type == 'mctn':
        start_mctn(train_loader, valid_loader, test_loader, param_mctn, device, epochs)


if __name__ == '__main__':
    main()

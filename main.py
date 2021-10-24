import argparse
import torch

from data_process import get_dataloaders
from parameters import param_mctn, param_mtt
from mctn_rnn.mctn import start_mctn
from mtt.mtt_cyclic import start_mtt_cyclic


def main():
    parser = argparse.ArgumentParser(description='CMU-MOSEI Sentiment Classifier')

    parser.add_argument('--model', type=str, default='mtt_cyclic',
                        help='Options are: mctn, mtt_cyclic, mtt_fuse')


    parser.add_argument('--dataset', type=str, default='mosei', help='Enter either mosei or mosi')
    args = parser.parse_args()

    model_type = str.lower(args.model.strip())


    dataset = str.lower(args.dataset.strip())
    if dataset == 'mosei':
        dataset = 'mosei_senti'

    seed = 7
    device = torch.device("cpu")
    torch.set_default_tensor_type('torch.FloatTensor')
    if torch.cuda.is_available():
        print("using cuda")
        torch.cuda.manual_seed(seed)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device("cuda")

    print(f'Processing dataset {dataset} for training on {model_type} model type')

    train_loader, valid_loader, test_loader = get_dataloaders(dataset)
    print("Loaded the Dataloaders")


    if model_type == 'mtt_cyclic':
        start_mtt_cyclic(train_loader, valid_loader, test_loader, param_mtt, device)
    elif model_type == 'mtt_fuse':
        print()
        #start_mtt_fuse(train_loader, valid_loader, test_loader, param_mtt, device)
    elif model_type == 'mctn':
        start_mctn(train_loader, valid_loader, test_loader, param_mctn, device)


if __name__ == '__main__':
    main()

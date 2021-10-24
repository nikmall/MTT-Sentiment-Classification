import time
from torch import optim
import torch

from mtt.modules_transformer import Encoder, Decoder, SentRegressor, Seq2SeqTransformer
from tools import epoch_time, init_weights, count_parameters
from score_metrics import mosei_scores


def start_mtt_cyclic(train_loader, valid_loader, test_loader, param_mtt, device):

    ENC_EMB_DIM = param_mtt['enc_emb_dim']
    DEC_EMB_DIM = param_mtt['dec_emb_dim']
    HID_DIM = param_mtt['hid_dim']  # same as text embedding
    ENC_LAYERS = param_mtt['enc_layers']
    DEC_LAYERS = param_mtt['dec_layers']
    ENC_HEADS = param_mtt['enc_heads']
    DEC_HEADS = param_mtt['dec_heads']
    ENC_PF_DIM = param_mtt['enc_pf_dim']
    DEC_PF_DIM = param_mtt['dec_pf_dim']
    ENC_DROPOUT = param_mtt['enc_dropout']
    DEC_DROPOUT = param_mtt['dec_dropout']

    MAX_LENGTH = param_mtt['max_length']

    enc = Encoder(ENC_EMB_DIM, HID_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, device, max_length=MAX_LENGTH)

    dec = Decoder(DEC_EMB_DIM, HID_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, device)

    SENT_HID_DIM = param_mtt['max_length']
    SENT_DROPOUT = param_mtt['max_length']
    SENT_N_LAYERS = param_mtt['sent_n_layers']
    SENT_FINAL_HID = param_mtt['sent_final_hid']

    N_EPOCHS = param_mtt['n_epochs']

    regression = SentRegressor(ENC_EMB_DIM, SENT_HID_DIM, SENT_FINAL_HID, SENT_N_LAYERS, SENT_DROPOUT)

    SRC_PAD_DIM = MAX_LENGTH
    TRG_PAD_DIM = MAX_LENGTH

    model = Seq2SeqTransformer(enc, dec, SRC_PAD_DIM, TRG_PAD_DIM, regression, device).to(device)

    model.apply(init_weights)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    init_lr = 0.0001
    optimizer = optim.Adam(model.parameters(), init_lr)
    criterion = torch.nn.MSELoss()

    train_model(model, train_loader, valid_loader, test_loader, optimizer, criterion, N_EPOCHS)

def train_model(model, train_loader, valid_loader, test_loader, optimizer, criterion, N_EPOCHS, params):
    best_valid_loss = float('inf')

    for epoch in range(0, N_EPOCHS):
        start_time = time.time()

        train_loss, pred_train, labels_train = train(model, train_loader, optimizer, criterion, params)
        valid_loss, pred_val, labels_val = evaluate(model, valid_loader, criterion, params)
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if epoch % 10 == 0:
            mosei_scores(pred_train, labels_train, message="Train Scores at epoch {}".format(epoch))
            mosei_scores(pred_val, labels_val, message="Val Scores at epoch {}".format(epoch))

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'mtt_cyclic.pt')

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.4f}%')
        print(f'\t Val. Loss: {valid_loss:.4f}%')

    # finally for test
    model.load_state_dict(torch.load('mtt_cyclic.pt'))

    test_loss, pred_test, labels_test = evaluate(model, test_loader, criterion, params)
    mosei_scores(pred_test, labels_test, message='Final Test Scores')


    print(f'Test Loss: {test_loss:.4f} ')



def train(model, train_loader, optimizer, criterion, params, clip=1):

    model.train()
    epoch_loss = 0
    preds = []
    truths = []

    for i_batch, (batch_X, batch_Y, batch_META) in enumerate(train_loader):
        sample_ind, text, audio, vision = batch_X

        src = text
        trg = audio
        label = batch_Y
        label = label.squeeze()

        optimizer.zero_grad()

        decoded, cycled_decoded, regression_score = model(src, trg, label)

        translate_loss = params['loss_dec_weight'] * criterion(decoded, trg)
        translate_cycle_loss = params['loss_dec_cycle_weight'] * criterion(cycled_decoded, src)
        translate_sent_loss = params['loss_regress_weight'] * criterion(regression_score, label)

        combined_loss = translate_loss + translate_cycle_loss + translate_sent_loss
        combined_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += combined_loss.item()

        preds.append(regression_score)
        truths.append(label)

    preds = torch.cat(preds)
    truths = torch.cat(truths)

    return epoch_loss / len(train_loader), preds, truths


def evaluate(model, valid_loader, criterion, params):

    model.eval()
    epoch_loss = 0
    preds = []
    truths = []

    with torch.no_grad():
        for i_batch, (batch_X, batch_Y, batch_META) in enumerate(valid_loader):
            sample_ind, text, audio, vision = batch_X

            src = text
            trg = audio
            label = batch_Y
            label = label.squeeze()

            decoded, cycled_decoded, regression_score = model(src, trg, label)

            translate_loss = params['loss_dec_weight'] * criterion(decoded, trg)
            translate_cycle_loss = params['loss_dec_cycle_weight'] * criterion(cycled_decoded, src)
            translate_sent_loss = params['loss_regress_weight'] * criterion(regression_score, label)

            combined_loss = translate_loss + translate_cycle_loss + translate_sent_loss
            epoch_loss += combined_loss.item()

            preds.append(regression_score)
            truths.append(label)

    preds = torch.cat(preds)
    truths = torch.cat(truths)

    return epoch_loss / len(valid_loader), preds, truths
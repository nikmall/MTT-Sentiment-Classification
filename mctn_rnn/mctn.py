import time
from torch import optim

from dataset import pad_modality
from mctn_rnn.modules_seq import Seq2Seq, Decoder, Encoder, SentRegressor, Attention
import torch

from tools import epoch_time, init_weights, count_parameters
from score_metrics import mosei_scores


def start_mctn(train_loader, valid_loader, test_loader, params, device, epochs):
    INPUT_DIM = train_loader.dataset.text.shape[1]
    OUTPUT_DIM = train_loader.dataset.text.shape[2]
    ENC_EMB_DIM = params["enc_emb_dim"]
    DEC_EMB_DIM = params["dec_emb_dim"]
    ENC_HID_DIM = params["enc_hid_dim"]
    DEC_HID_DIM = params["dec_hid_dim"]
    ENC_DROPOUT = params["enc_dropout"]
    DEC_DROPOUT = params["dec_dropout"]

    SENT_HID_DIM = params["sent_hid_dim"]
    SENT_DROPOUT = params["sent_dropout"]

    N_LAYERS = params["n_layers"]
    N_EPOCHS = epochs if epochs is not None else params["n_epochs"]

    attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

    regression = SentRegressor(ENC_HID_DIM, SENT_HID_DIM, OUTPUT_DIM, N_LAYERS, SENT_DROPOUT)

    model = Seq2Seq(enc, dec, regression, device).to(device)

    model.apply(init_weights)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    init_lr = 0.0001
    optimizer = optim.Adam(model.parameters(), init_lr)
    criterion = torch.nn.MSELoss()

    train_model(model, train_loader, valid_loader, test_loader, optimizer, criterion, N_EPOCHS, params, device)

def train_model(model, train_loader, valid_loader, test_loader, optimizer, criterion, N_EPOCHS, params, device):

    best_valid_loss = float('inf')

    for epoch in range(0, N_EPOCHS):
        start_time = time.time()

        train_loss, pred_train, labels_train = train(model, train_loader, optimizer, criterion, params, device)
        valid_loss, pred_val, labels_val = evaluate(model, valid_loader, criterion, params, device)
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if epoch % 10 == 0:
            mosei_scores(pred_train, labels_train, message="Train Scores at epoch {}".format(epoch))
            mosei_scores(pred_val, labels_val, message="Val Scores at epoch {}".format(epoch))

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'mctn_rnn.pt')

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.4f}%')
        print(f'\t Val. Loss: {valid_loss:.4f}%')

    # finally for test
    model.load_state_dict(torch.load('mctn_rnn.pt', map_location=device))

    test_loss, pred_test, labels_test = evaluate(model, test_loader, criterion, device)

    mosei_scores(pred_test, labels_test, message='Final Test Scores')
    print(f'Test Loss: {test_loss:.4f} ')


def train(model, train_loader, optimizer, criterion, params, device, clip=1):
    model.train()
    epoch_loss = 0
    preds = []
    truths = []

    for i_batch, (batch_X, batch_Y, batch_META) in enumerate(train_loader):
        sample_ind, text, audio, vision = batch_X

        audio = pad_modality(audio, text.shape[2], audio.shape[2])
        text = text.to(device=device)
        audio = audio.to(device=device)
        vision = vision.to(device=device)

        src = text.permute(1, 0, 2)
        src = src.to(device=device)
        trg = audio.permute(1, 0, 2)
        trg = trg.to(device=device)

        label = batch_Y.permute(1, 0, 2)
        label = label.squeeze(0).to(device=device)


        optimizer.zero_grad()

        decoded, cycled_decoded, regression_score = model(src, trg, label, device)

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


def evaluate(model, valid_loader, criterion, params, device):
    model.eval()
    epoch_loss = 0
    preds = []
    truths = []

    with torch.no_grad():
        for i_batch, (batch_X, batch_Y, batch_META) in enumerate(valid_loader):
            sample_ind, text, audio, vision = batch_X

            audio = pad_modality(audio, text.shape[2], audio.shape[2])
            text = text.to(device=device)
            audio = audio.to(device=device)
            vision = vision.to(device=device)

            src = text.permute(1, 0, 2)
            src = src.to(device=device)
            trg = audio.permute(1, 0, 2)
            trg = trg.to(device=device)

            label = batch_Y.permute(1, 0, 2)
            label = label.squeeze(0).to(device=device)

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


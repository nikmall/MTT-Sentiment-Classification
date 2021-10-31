import time
from torch import optim
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from mtt_fuse.modules_transformer_fuse import Encoder, Decoder, SentRegressor, Seq2SeqTransformer
from tools import epoch_time, init_weights, count_parameters
from score_metrics import mosei_scores


def start_mtt_fuse(train_loader, valid_loader, test_loader, param_mtt, device, epochs):

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

    MAX_LENGTH = train_loader.dataset.text.shape[1]

    enc_text = Encoder(ENC_EMB_DIM, HID_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, device, MAX_LENGTH)
    dec_audio = Decoder(DEC_EMB_DIM, HID_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, device, MAX_LENGTH)

    enc_audio = Encoder(ENC_EMB_DIM, HID_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, device, MAX_LENGTH)
    dec_text = Decoder(DEC_EMB_DIM, HID_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, device, MAX_LENGTH)

    SENT_HID_DIM = param_mtt['sent_hid_dim']
    SENT_FINAL_HID = param_mtt['sent_final_hid']
    SENT_N_LAYERS = param_mtt['sent_n_layers']
    SENT_DROPOUT = param_mtt['sent_dropout']

    N_EPOCHS = epochs if epochs is not None else param_mtt['n_epochs']

    regression = SentRegressor(ENC_EMB_DIM, SENT_HID_DIM, SENT_FINAL_HID, SENT_N_LAYERS, SENT_DROPOUT)

    SRC_PAD_DIM = ENC_EMB_DIM
    TRG_PAD_DIM = DEC_EMB_DIM

    model = Seq2SeqTransformer(enc_text, dec_audio, enc_audio, dec_text, SRC_PAD_DIM, TRG_PAD_DIM, regression, device).to(device)
    print(model)
    model.apply(init_weights)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    init_lr = 0.001
    min_lr = 0.0001
    optimizer = optim.Adam(model.parameters(), init_lr)
    criterion = torch.nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=param_mtt['lr_patience'], min_lr=min_lr,
                                  factor=0.1, verbose=True)


    train_model(model, train_loader, valid_loader, test_loader, optimizer, criterion, N_EPOCHS, param_mtt, scheduler, device)


def train_model(model, train_loader, valid_loader, test_loader, optimizer, criterion, N_EPOCHS, params, scheduler, device):
    best_valid_loss = float('inf')

    for epoch in range(0, N_EPOCHS):
        start_time = time.time()

        train_loss, pred_train, labels_train = train(model, train_loader, optimizer, criterion, params, device)
        valid_loss, pred_val, labels_val = evaluate(model, valid_loader, criterion, params, device)
        scheduler.step(valid_loss)
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if epoch % 10 == 0 and epoch > 0:
            mosei_scores(pred_train, labels_train, message="Train Scores at epoch {}".format(epoch))
            mosei_scores(pred_val, labels_val, message="Val Scores at epoch {}".format(epoch))

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'mtt_fuse.pt')

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.4f}%')
        print(f'\t Val. Loss: {valid_loss:.4f}%')

    # finally for test
    print("DEVICE: ", device)
    model.load_state_dict(torch.load('mtt_fuse.pt', map_location=device))

    test_loss, pred_test, labels_test = evaluate(model, test_loader, criterion, params, device)
    mosei_scores(pred_test, labels_test, message='Final Test Scores')


    print(f'Test Loss: {test_loss:.4f} ')



def train(model, train_loader, optimizer, criterion, params, device, clip=1):

    model.train()
    epoch_loss = 0
    preds = []
    truths = []

    for i_batch, (batch_X, batch_Y, batch_META) in enumerate(train_loader):
        sample_ind, text, audio, vision = batch_X

        text = text.to(device=device)
        audio = audio.to(device=device)
        label = batch_Y
        label = label.squeeze().to(device=device)

        optimizer.zero_grad()

        output_audio, output_text, regression_score = model(text, audio, label)

        translate_loss = params['loss_dec_weight'] * criterion(output_audio, audio)
        translate_cycle_loss = params['loss_dec_cycle_weight'] * criterion(output_text, text)
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

            text = text.to(device=device)
            audio = audio.to(device=device)
            label = batch_Y
            label = label.squeeze().to(device=device)

            output_audio, output_text, regression_score = model(text, audio, label)

            translate_loss = params['loss_dec_weight'] * criterion(output_audio, audio)
            translate_cycle_loss = params['loss_dec_cycle_weight'] * criterion(output_text, text)
            translate_sent_loss = params['loss_regress_weight'] * criterion(regression_score, label)

            combined_loss = translate_loss + translate_cycle_loss + translate_sent_loss
            epoch_loss += combined_loss.item()

            preds.append(regression_score)
            truths.append(label)

    preds = torch.cat(preds)
    truths = torch.cat(truths)

    return epoch_loss / len(valid_loader), preds, truths
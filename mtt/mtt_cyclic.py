import time
from torch import optim
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from mtt.modules_transformer import Encoder, Decoder, SentRegressor, SentRegressorRNN, Seq2SeqTransformer, Seq2SeqTransformerRNN
from tools import epoch_time, init_weights, count_parameters
from score_metrics import mosei_scores
from dataset import pad_modality


def start_mtt_cyclic(train_loader, valid_loader, test_loader, param_mtt, device, epochs):

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

    DED_HID_DIM_2 = train_loader.dataset.vision.shape[2] + 1

    MAX_LENGTH_ENC = 50
    MAX_LENGTH_DEC = 50

    enc = Encoder(ENC_EMB_DIM, HID_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, device, MAX_LENGTH_ENC)

    dec = Decoder(DEC_EMB_DIM, HID_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, device, MAX_LENGTH_DEC, ENC_EMB_DIM, ENC_EMB_DIM)

    enc2 = Encoder(param_mtt['enc_emb_dim'], param_mtt['hid_dim'], param_mtt['enc_layers_2'], param_mtt['enc_heads_2'],
                   param_mtt['enc_pf_dim_2'], ENC_DROPOUT, device, MAX_LENGTH_ENC)

    dec2 = Decoder(DED_HID_DIM_2, DED_HID_DIM_2,  param_mtt['dec_layers_2'], param_mtt['dec_heads_2'],
                   param_mtt['dec_pf_dim_2'], DEC_DROPOUT, device, MAX_LENGTH_DEC, ENC_EMB_DIM, ENC_EMB_DIM)

    SENT_HID_DIM = param_mtt['sent_hid_dim']
    SENT_FINAL_HID = param_mtt['sent_final_hid']
    SENT_N_LAYERS = param_mtt['sent_n_layers']
    SENT_DROPOUT = param_mtt['sent_dropout']
    BIDIRECT = param_mtt['bidirect']

    N_EPOCHS = epochs if epochs is not None else param_mtt['n_epochs']
    SRC_PAD_DIM = ENC_EMB_DIM
    TRG_PAD_DIM = DEC_EMB_DIM

    # Transformer only
    if param_mtt['transformer_regression'] == True:
        encoder_sent = Encoder(ENC_EMB_DIM, HID_DIM, 2, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, device, MAX_LENGTH_ENC)
        regression = SentRegressor(ENC_EMB_DIM, SENT_HID_DIM, SENT_FINAL_HID, SENT_N_LAYERS, device)
        model = Seq2SeqTransformer(enc, dec, SRC_PAD_DIM, TRG_PAD_DIM, regression, encoder_sent, device).to(device)

    else:
        # LSTM
        regression = SentRegressorRNN(ENC_EMB_DIM, SENT_HID_DIM, SENT_FINAL_HID, SENT_N_LAYERS, SENT_DROPOUT, BIDIRECT)
        model = Seq2SeqTransformerRNN(enc, dec, enc2, dec2, SRC_PAD_DIM, TRG_PAD_DIM, regression, device).to(device)

    print(model)

    model.apply(init_weights)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    init_lr = 0.0001
    min_lr = 0.0001
    optimizer = optim.Adam(model.parameters(), init_lr)
    criter_tran = torch.nn.MSELoss()
    criter_regr = torch.nn.MSELoss()
    criterion = (criter_tran, criter_regr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=param_mtt['lr_patience'], min_lr=min_lr,
                                  factor=0.1, verbose=True)

    f1_score = train_model(model, train_loader, valid_loader, test_loader, optimizer, criterion, N_EPOCHS, param_mtt,
                           scheduler, device)
    return f1_score


def train_model(model, train_loader, valid_loader, test_loader, optimizer, criterion, N_EPOCHS, params, scheduler, device):
    best_valid_loss = float('inf')

    for epoch in range(1, N_EPOCHS+1):
        start_time = time.time()

        train_loss, pred_train, labels_train = train(model, train_loader, optimizer, criterion, params, device)
        valid_loss, pred_val, labels_val = evaluate(model, valid_loader, criterion, params, device)
        # scheduler.step(valid_loss)
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f'Epoch: {epoch:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.4f}%')
        print(f'\t Val. Loss: {valid_loss:.4f}%')

        if epoch % 10 == 0 and epoch > 0:
            mosei_scores(pred_train, labels_train, message="Train Scores at epoch {}".format(epoch))
            mosei_scores(pred_val, labels_val, message="Val Scores at epoch {}".format(epoch))

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'mtt_cyclic.pt')

    test_loss, pred_test, labels_test = evaluate(model, test_loader, criterion, params, device)
    mosei_scores(pred_test, labels_test, message='On current epoch model -  Test Scores')
    print('')

    model.load_state_dict(torch.load('mtt_cyclic.pt', map_location=device))

    test_loss, pred_test, labels_test = evaluate(model, test_loader, criterion, params, device)
    f1_score = mosei_scores(pred_test, labels_test, message='Final Test Scores on best val error model')

    print(f'Minimum validation loss overall is {train_loss}')
    print(f'Test Loss: {test_loss:.4f} ')

    return f1_score


def train(model, train_loader, optimizer, criterion, params, device, clip=10):

    model.train()
    epoch_loss = 0
    preds = []
    truths = []

    for i_batch, (batch_X, batch_Y, batch_META) in enumerate(train_loader):
        sample_ind, text, audio, vision = batch_X

        src = text.to(device=device)

        if params["fuse_modalities"]:
            fused_a_v = torch.cat((audio, vision), dim=2)
            if params["cyclic"]:
                fused_a_v = pad_modality(fused_a_v, text.shape[2], fused_a_v.shape[2])
            else:
                fused_a_v = pad_modality(fused_a_v, fused_a_v.shape[2] + 2, fused_a_v.shape[2]) # pad with 1 for divisions
            trg = fused_a_v
        else:
            if params["cyclic"]:
                trg1 = pad_modality(audio, text.shape[2], audio.shape[2])
            else:
                trg1 = pad_modality(audio, audio.shape[2] + 1, audio.shape[2]) # pad with 1 for divisions

            trg2 = pad_modality(vision, vision.shape[2] + 1, vision.shape[2])

        trg1 = trg1.to(device=device)
        trg2 = trg2.to(device=device)

        label = batch_Y
        label = label.squeeze().to(device=device)

        optimizer.zero_grad()

        decoded, cycled_decoded, decoded_2, regression_score = model(src, trg1, trg2, label)

        criter_tran = criterion[0]
        criter_regr = criterion[1]
        translate_loss = params['loss_dec_weight'] * criter_tran(decoded, trg1)
        translate_cycle_loss = params['loss_dec_cycle_weight'] * criter_tran(cycled_decoded, src)
        translate_loss_2 = params['loss_dec_weight'] * criter_tran(decoded_2, trg2)
        translate_sent_loss = params['loss_regress_weight'] * criter_regr(regression_score, label)

        combined_loss = translate_loss + translate_cycle_loss + translate_sent_loss + translate_loss_2
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

            src = text.to(device=device)

            if params["fuse_modalities"]:
                fused_a_v = torch.cat((audio, vision), dim=2)
                if params["cyclic"]:
                    fused_a_v = pad_modality(fused_a_v, text.shape[2], fused_a_v.shape[2])
                else:
                    fused_a_v = pad_modality(fused_a_v, fused_a_v.shape[2] + 2,
                                             fused_a_v.shape[2])  # pad with 1 for divisions
                trg = fused_a_v
            else:
                if params["cyclic"]:
                    trg1 = pad_modality(audio, text.shape[2], audio.shape[2])
                else:
                    trg1 = pad_modality(audio, audio.shape[2] + 1, audio.shape[2])  # pad with 1 for divisions

                trg2 = pad_modality(vision, vision.shape[2] + 1, vision.shape[2])

            # trg = trg.to(device=device)
            trg1 = trg1.to(device=device)
            trg2 = trg2.to(device=device)

            label = batch_Y
            label = label.squeeze().to(device=device)

            decoded, cycled_decoded, decoded_2, regression_score = model(src, trg1, trg2, label)

            criter_tran = criterion[0]
            criter_regr = criterion[1]
            translate_loss = params['loss_dec_weight'] * criter_tran(decoded, trg1)
            translate_cycle_loss = params['loss_dec_cycle_weight'] * criter_tran(cycled_decoded, src)
            translate_loss_2 = params['loss_dec_weight'] * criter_tran(decoded_2, trg2)
            translate_sent_loss = params['loss_regress_weight'] * criter_regr(regression_score, label)

            combined_loss = translate_loss + translate_cycle_loss + translate_sent_loss + translate_loss_2
            epoch_loss += combined_loss.item()

            preds.append(regression_score)
            truths.append(label)

    preds = torch.cat(preds)
    truths = torch.cat(truths)

    return epoch_loss / len(valid_loader), preds, truths

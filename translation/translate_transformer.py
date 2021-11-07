from __future__ import unicode_literals, print_function, division
import random
import sys
from io import open
import unicodedata
import re
import string
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from exp_dataset import get_dataloaders
import torch
import numpy as np
import time
import math
import config as configs
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False

class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, max_length=100):
        super().__init__()

        self.device = device

        # self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, input_dim)  # (max_length, hid_dim)

        self.layers = nn.ModuleList([EncoderLayer(hid_dim, n_heads, pf_dim, dropout, device) for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src, src_mask):
        # src = [batch size, src len]
        # src_mask = [batch size, 1, 1, src len]

        batch_size = src.shape[0]
        src_len = src.shape[1]

        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        # pos = [batch size, src len]

        # src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        src = self.dropout((src * self.scale) + self.pos_embedding(pos))

        # src = [batch size, src len, hid dim]

        for layer in self.layers:
            src = layer(src, src_mask)

        # src = [batch size, src len, hid dim]

        return src


class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()

        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)

        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # src = [batch size, src len, hid dim]
        # src_mask = [batch size, 1, 1, src len]

        # self attention
        _src, _ = self.self_attention(src, src, src, src_mask)

        # dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))

        # src = [batch size, src len, hid dim]

        # positionwise feedforward
        _src = self.positionwise_feedforward(src)

        # dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))

        # src = [batch size, src len, hid dim]

        return src


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        # Q = [batch size, query len, hid dim]
        # K = [batch size, key len, hid dim]
        # V = [batch size, value len, hid dim]

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Q = [batch size, n heads, query len, head dim]
        # K = [batch size, n heads, key len, head dim]
        # V = [batch size, n heads, value len, head dim]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # energy = [batch size, n heads, query len, key len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)
        # attention = [batch size, n heads, query len, key len]

        x = torch.matmul(self.dropout(attention), V)
        # x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()
        # x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.hid_dim)
        # x = [batch size, query len + n_heads(ss ??), hid dim]

        x = self.fc_o(x)
        # x = [batch size, query len, hid dim]

        return x, attention


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, seq len, hid dim]

        x = self.dropout(torch.relu(self.fc_1(x)))
        # x = [batch size, seq len, pf dim]

        x = self.fc_2(x)
        # x = [batch size, seq len, hid dim]

        return x


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]

        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        # repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # hidden = [batch size, src len, dec hid dim]
        # encoder_outputs = [batch size, src len, enc hid dim * 2]

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        # energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)

        # attention= [batch size, src len]

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, max_length=100):
        super().__init__()

        self.device = device
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([DecoderLayer(hid_dim, n_heads, pf_dim, dropout, device)
                                     for _ in range(n_layers)])

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg = [batch size, trg len]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, 1, trg len, trg len]
        # src_mask = [batch size, 1, 1, src len]

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        # pos = [batch size, trg len]

        trg = self.dropout((trg * self.scale) + self.pos_embedding(pos))
        # trg = [batch size, trg len, hid dim]

        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)

        # trg = [batch size, trg len, hid dim]
        # attention = [batch size, n heads, trg len, src len]

        output = self.fc_out(trg)

        # output = [batch size, trg len, output dim]

        return output, attention


class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)

        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)

        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg = [batch size, trg len, hid dim]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, 1, trg len, trg len]
        # src_mask = [batch size, 1, 1, src len]

        # self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)

        # dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
        # trg = [batch size, trg len, hid dim]

        # encoder attention
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)

        # dropout, residual connection and layer norm
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
        # trg = [batch size, trg len, hid dim]

        # position-wise feedforward
        _trg = self.positionwise_feedforward(trg)

        # dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]

        return trg, attention


class SentRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout, bidirect=False):
        super(SentRegressor, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, bidirectional=bidirect, dropout=dropout,
                            num_layers=self.n_layers, batch_first=True)
        if bidirect:
            self.fc = nn.Linear(self.hidden_dim * self.n_layers * 2, self.output_dim)
        else:
            self.fc = nn.Linear(self.hidden_dim * self.n_layers, self.output_dim)
        self.fc2 = nn.Linear(self.output_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoded):
        output, (hidden, cell) = self.lstm(encoded)
        if self.n_layers == 2:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = hidden.squeeze()
            hidden = self.dropout(hidden)
        fc_out = F.relu(self.fc(hidden))
        final_out = self.fc2(fc_out)

        return final_out.squeeze()


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_dim, trg_pad_dim, regression, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_dim = src_pad_dim
        self.trg_pad_dim = trg_pad_dim
        self.device = device
        self.regression = regression

    # mask for pre-trained embedding inputs (3dim)
    def make_src_mask(self, src):
        # src = [batch size, src len, dim]

        src_pad = torch.zeros(src.shape[0], src.shape[1], self.src_pad_dim, device=self.device)

        src_mask = torch.all(torch.eq(src, src_pad), axis=2)

        src_mask = src_mask.unsqueeze(1).unsqueeze(2)
        # src_mask = [batch size, 1, 1, src len]

        return src_mask

    def make_trg_mask(self, trg):

        # trg = [batch size, trg len]

        trg_pad = torch.zeros(trg.shape[0], trg.shape[1], self.trg_pad_dim, device=self.device)

        trg_pad_mask = torch.all(torch.eq(trg, trg_pad), axis=2).unsqueeze(1).unsqueeze(2)
        # trg_pad_mask = [batch size, 1, 1, trg len]

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
        # trg_sub_mask = [trg len, trg len]

        trg_mask = trg_pad_mask & trg_sub_mask
        # trg_mask = [batch size, 1, trg len, trg len]

        return trg_mask


    def forward(self, src, trg, label, teacher_forcing_ratio=0.5):
        #src = [batch size, src len, dim]
        #trg = [batch size, trg len, dim]

        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        #src_mask = [batch size, 1, 1, src len]
        #trg_mask = [batch size, 1, trg len, trg len]

        enc_src = self.encoder(src, src_mask)
        #enc_src = [batch size, src len, hid dim]

        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        #output = [batch size, trg len, output dim]
        #attention = [batch size, n heads, trg len, src len]

        src_mask_2 = self.make_src_mask(output)

        enc_src_2 = self.encoder(output, src_mask_2)

        trg_mask_2 = self.make_trg_mask(enc_src_2)

        output_2, attention_2 = self.decoder(src, enc_src_2, trg_mask_2, src_mask)

        regression_score = self.regression(enc_src)

        return output, output_2, regression_score


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(model, train_loader, optimizer, criterion, clip=1):

    model.train()
    epoch_loss = 0
    preds = []
    truths = []

    for i_batch, (batch_X, batch_Y, batch_META) in enumerate(train_loader):
        sample_ind, text, audio, vision = batch_X

        src = text.to(device=device)
        trg = audio.to(device=device)
        label = batch_Y.to(device=device)
        label = label.squeeze()

        optimizer.zero_grad()

        decoded, cycled_decoded, regression_score = model(src, trg, label)

        translate_loss = configs.conf['loss_dec_weight'] * criterion(decoded, trg)
        translate_cycle_loss = configs.conf['loss_dec_cycle_weight'] * criterion(cycled_decoded, src)
        translate_sent_loss = configs.conf['loss_regress_weight'] * criterion(regression_score, label)

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


def evaluate(model, valid_loader, criterion):

    model.eval()
    epoch_loss = 0
    preds = []
    truths = []

    with torch.no_grad():
        for i_batch, (batch_X, batch_Y, batch_META) in enumerate(valid_loader):
            sample_ind, text, audio, vision = batch_X

            src = text.to(device=device)
            trg = audio.to(device=device)
            label = batch_Y.to(device=device)
            label = label.squeeze()

            decoded, cycled_decoded, regression_score = model(src, trg, label)

            translate_loss = configs.conf['loss_dec_weight'] * criterion(decoded, trg)
            translate_cycle_loss = configs.conf['loss_dec_cycle_weight'] * criterion(cycled_decoded, src)
            translate_sent_loss = configs.conf['loss_regress_weight'] * criterion(regression_score, label)

            combined_loss = translate_loss + translate_cycle_loss + translate_sent_loss
            epoch_loss += combined_loss.item()

            preds.append(regression_score)
            truths.append(label)

    preds = torch.cat(preds)
    truths = torch.cat(truths)

    return epoch_loss / len(valid_loader), preds, truths


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def start():
    train_loader, valid_loader, test_loader = get_dataloaders()

    ENC_EMB_DIM = train_loader.dataset.text.shape[2]  # train_loader.dataset.text.shape[1]  # 300
    DEC_EMB_DIM = train_loader.dataset.text.shape[2]  # train_loader.dataset.text.shape[2]

    HID_DIM = 300  # == 300 the text embedding  # was 256 in code
    ENC_LAYERS = 3
    DEC_LAYERS = 3

    ENC_HEADS = 5  # 8
    DEC_HEADS = 5  # 8
    ENC_PF_DIM = 512
    DEC_PF_DIM = 512

    ENC_DROPOUT = 0.3
    DEC_DROPOUT = 0.3

    MAX_LENGTH = train_loader.dataset.text.shape[1]

    enc = Encoder(ENC_EMB_DIM, HID_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, device, max_length=MAX_LENGTH)

    dec = Decoder(DEC_EMB_DIM, HID_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, device, max_length=MAX_LENGTH)

    SENT_HID_DIM = 192
    SENT_DROPOUT = 0.4

    SENT_N_LAYERS = 2 # 2
    N_EPOCHS = 50 # 100
    SENT_FINAL_HID = 128

    regression = SentRegressor(ENC_EMB_DIM, SENT_HID_DIM, SENT_FINAL_HID, SENT_N_LAYERS, SENT_DROPOUT)

    SRC_PAD_DIM = 300
    TRG_PAD_DIM = 300

    model = Seq2Seq(enc, dec, SRC_PAD_DIM, TRG_PAD_DIM, regression, device).to(device)
    print(model)
    model.apply(init_weights)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    init_lr = 0.0001
    optimizer = optim.Adam(model.parameters(), init_lr)
    criterion = torch.nn.MSELoss()

    train_model(model, train_loader, valid_loader, test_loader, optimizer, criterion, N_EPOCHS)


def train_model(model, train_loader, valid_loader, test_loader, optimizer, criterion, N_EPOCHS):  # CLIP = 1
    best_valid_loss = float('inf')

    for epoch in range(1, N_EPOCHS + 1):
        start_time = time.time()

        train_loss, pred_train, labels_train = train(model, train_loader, optimizer, criterion)
        valid_loss, pred_val, labels_val = evaluate(model, valid_loader, criterion)
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if epoch % 10 == 0:
            get_preds_statistics(pred_train, labels_train, message="Train Scores at epoch {}".format(epoch))
            get_preds_statistics(pred_val, labels_val, message="Val Scores at epoch {}".format(epoch))

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'mctn_trans.pt')

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.4f}%')
        print(f'\t Val. Loss: {valid_loss:.4f}%')

    # finally for test
    model.load_state_dict(torch.load('mctn_trans.pt'))

    test_loss, pred_test, labels_test = evaluate(model, test_loader, criterion)
    get_preds_statistics(pred_test, labels_test, message="Test Final Scores: ")
    mosei_scores(pred_test, labels_test, message="Test Scores Final 2 -mosei_scores ")

    print(f'Test Loss: {test_loss:.4f} ')


def mosei_scores(pred, labels, message=""):
    print(message)
    test_preds = pred.view(-1).cpu().detach().numpy()
    test_truth = labels.view(-1).cpu().detach().numpy()

    mae = np.mean(np.absolute(test_preds - test_truth))
    corr = np.corrcoef(test_preds, test_truth)[0][1]

    print("MAE: ", mae)
    print("Correlation Coefficient: ", corr)

    exclude_zero = True
    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0 or (not exclude_zero)])
    binary_truth = (test_truth[non_zeros] > 0)
    binary_preds = (test_preds[non_zeros] > 0)
    f_score = f1_score(binary_truth, binary_preds, average='weighted')
    print("Accuracy: ", accuracy_score(binary_truth, binary_preds))
    print("F1 score: ", f_score)


def get_preds_statistics(predictions, y_test, message):
    """
    Get MAE, CORR and Accuracy

    Args:
      predictions:
      y_test: labels

    """
    predictions = predictions.view(-1).cpu().detach().numpy()
    y_test = y_test.view(-1).cpu().detach().numpy()

    print(message)
    mae = np.mean(np.absolute(predictions - y_test))
    print("mae: {}".format(mae))
    corr = np.corrcoef(predictions, y_test)[0][1]
    print("corr: {}".format(corr))
    mult = round(
        sum(np.round(predictions) == np.round(y_test)) / float(len(y_test)), 5)
    print("mult_acc: {}".format(mult))
    f_score = round(f1_score(np.round(predictions),
                             np.round(y_test), average='weighted'),
                    5)
    print("mult f_score: {}".format(f_score))
    true_label = (y_test >= 0)
    predicted_label = (predictions >= 0)
    print("Confusion Matrix :")
    print(confusion_matrix(true_label, predicted_label))
    print("Classification Report :")
    print(classification_report(true_label, predicted_label, digits=5))
    print("Accuracy: {}".format(accuracy_score(true_label, predicted_label)))
    sys.stdout.flush()


if __name__ == "__main__":
    start()

import random
import torch.nn as nn
import torch.nn.functional as F
import torch

"""
This seems to work for only 1-layer
"""
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, n_layers=1):
        super().__init__()

        self.enc_hid_dim = enc_hid_dim
        self.n_layers = n_layers

        self.rnn = nn.LSTM(emb_dim, enc_hid_dim, n_layers, bidirectional=True)

        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.fc2 = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        dropped = self.dropout(src)
        # embedded = [src len, batch size, emb dim]

        outputs, (hidden, cell) = self.rnn(dropped)

        # outputs are always from the top hidden layer
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))

        cell = torch.tanh(self.fc2(torch.cat((cell[-2, :, :], cell[-1, :, :]), dim=1)))

        return outputs, hidden, cell


"""
Note Bahdanau  attention -  content based function is concat
"""
class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):

        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        # repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        attention = self.v(energy).squeeze(2)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention, n_layers=1):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention
        self.emb_dim = emb_dim
        self.n_layers = n_layers

        self.rnn = nn.LSTM((enc_hid_dim * 2) + emb_dim, dec_hid_dim, n_layers)

        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs):

        # n directions in the decoder will both always be 1, therefore:
        input = input.unsqueeze(0)

        embedded = self.dropout(input)

        a = self.attention(hidden, encoder_outputs)

        a = a.unsqueeze(1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        weighted = torch.bmm(a, encoder_outputs)

        weighted = weighted.permute(1, 0, 2)

        rnn_input = torch.cat((embedded, weighted), dim=2)

        output, (hidden, cell) = self.rnn(rnn_input, (hidden.unsqueeze(0), cell.unsqueeze(0)))

        assert (output == hidden).all()

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))

        return prediction, hidden.squeeze(0), cell.squeeze(0)


class SentRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout):
        super(SentRegressor, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.lstm = nn.LSTM(self.input_dim * 2, self.hidden_dim, bidirectional=False, dropout=dropout,
                            num_layers=n_layers)  # tanh is default

        self.fc = nn.Linear(self.hidden_dim * 2, self.output_dim)  # *2 when bidirectional as we get 2 hidden out
        self.fc2 = nn.Linear(self.output_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoded):
        output, (hidden, cell) = self.lstm(encoded)
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        fc_out = F.relu(self.fc(hidden))
        final_out = self.fc2(fc_out)

        return final_out.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, regression, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.regression = regression

    def forward(self, src, trg, label, teacher_forcing_ratio=0.5):
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        src_len = src.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder decoder_outputs
        decoder_outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        encoder_outputs, hidden, cell = self.encoder(src)


        # first input to the decoder is just zeros
        input = torch.zeros(batch_size, self.decoder.emb_dim).to(device=device)

        for t in range(0, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)

            decoder_outputs[t] = output

            #  teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # if teacher forcing, use actual next token as next input
            # if not, use predicted output token
            input = trg[t] if teacher_force else output

        # Cycle - 2nd iteration for cycling
        encoder_outputs_2, hidden_2, cell_2 = self.encoder(decoder_outputs)  # decoder_outputs is audio

        input2 = torch.zeros(batch_size, self.decoder.emb_dim)
        decoder_outputs_2 = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        for t in range(0, src_len):
            output, hidden_2, cell_2 = self.decoder(input2, hidden_2, cell_2, encoder_outputs_2)  # should return text

            decoder_outputs_2[t] = output

            teacher_force = random.random() < teacher_forcing_ratio
            input2 = src[t] if teacher_force else output

        regression_score = self.regression(encoder_outputs)

        return decoder_outputs, decoder_outputs_2, regression_score

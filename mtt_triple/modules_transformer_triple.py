from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch
import math

class Encoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device,
                 max_length, kdim, vdim, dropout_att):
        super().__init__()

        self.device = device
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, n_heads, pf_dim, dropout, kdim, vdim, dropout_att,
                                                  device) for _ in range(n_layers)])

        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src, src_mask):
        batch_size = src.shape[0]
        src_len = src.shape[1]

        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        src = self.dropout((src * self.scale) + self.pos_embedding(pos))


        for layer in self.layers:
            src = layer(src, src_mask)

        output = self.fc_out(src)

        return output


class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, kdim, vdim, dropout_att, device):
        super().__init__()

        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        # self.self_attention = nn.MultiheadAttention(embed_dim=hid_dim, num_heads=n_heads, kdim=kdim,
        #                                            vdim=vdim, batch_first=True, dropout=dropout_att)
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        _src, _ = self.self_attention(src, src, src, src_mask)

        src = self.self_attn_layer_norm(src + self.dropout(_src))

        _src = self.positionwise_feedforward(src)

        src = self.ff_layer_norm(src + self.dropout(_src))

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

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)

        x = torch.matmul(self.dropout(attention), V)

        x = x.permute(0, 2, 1, 3).contiguous()

        x = x.view(batch_size, -1, self.hid_dim)

        x = self.fc_o(x)

        return x, attention


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout, out_hid_dim=''):
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(torch.relu(self.fc_1(x)))

        x = self.fc_2(x)

        return x


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        attention = self.v(energy).squeeze(2)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, enc_dim, n_layers, n_heads, pf_dim, dropout, device,
                 max_length, kdim, vdim, dropout_att):
        super().__init__()

        self.device = device

        self.pos_embedding = nn.Embedding(max_length, output_dim)

        self.layers = nn.ModuleList([DecoderLayer(output_dim, enc_dim, n_heads, pf_dim, dropout, device, kdim, vdim, dropout_att)
                                     for _ in range(n_layers)])

        # self.fc_out = nn.Linear(output_dim, enc_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([enc_dim])).to(device)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        trg = self.dropout((trg * self.scale) + self.pos_embedding(pos))

        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)

        # output = self.fc_out(trg)
        output = trg

        return output, attention


class DecoderLayer(nn.Module):
    def __init__(self, output_dim, enc_dim, n_heads, pf_dim, dropout, device, kdim, vdim, dropout_att):
        super().__init__()
        self.n_heads = n_heads
        self.self_attn_layer_norm = nn.LayerNorm(output_dim)
        # self.enc_attn_layer_norm = nn.LayerNorm(enc_dim)
        self.ff_layer_norm = nn.LayerNorm(output_dim)

        # self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.self_attention = nn.MultiheadAttention(embed_dim=output_dim, num_heads=n_heads,
                                                    batch_first=True, dropout=dropout_att)
        self.encoder_attention = nn.MultiheadAttention(embed_dim=output_dim, num_heads=n_heads, kdim=enc_dim,
                                                        vdim=enc_dim, batch_first=True, dropout=dropout_att)
        # self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(output_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        _trg, _ = self.self_attention(trg, trg, trg) # , trg_mask)

        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))

        # _trg, attention = self.encoder_attention(trg, enc_src, enc_src, key_padding_mask=src_mask.squeeze())
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src) # , src_mask)
        attention = 0
        # trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))

        _trg = self.positionwise_feedforward(trg)

        trg = self.ff_layer_norm(trg + self.dropout(_trg))

        return trg, attention


class SentRegressorRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout, bidirect=False):
        super(SentRegressorRNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.bidirect = bidirect

        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, bidirectional=self.bidirect,
                            dropout=dropout,
                            num_layers=self.n_layers, batch_first=True)
        if self.bidirect:
            self.fc = nn.Linear(self.hidden_dim * 2, self.output_dim)
        else:
            self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        self.fc2 = nn.Linear(self.output_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoded):
        output, (hidden, cell) = self.lstm(encoded)
        if self.bidirect:
            hidden_out = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden_out = self.dropout(hidden[-1, :, :])
        fc_out = F.relu(self.fc(hidden_out))
        final_out = self.fc2(fc_out)

        return final_out.squeeze()


class Seq2SeqTransformerRNN(nn.Module):
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
        src_pad = torch.zeros(src.shape[0], src.shape[1], self.src_pad_dim, device=self.device)

        src_mask = torch.all(torch.eq(src, src_pad), axis=2)

        src_mask = src_mask.unsqueeze(1).unsqueeze(2)

        return src_mask

    def make_trg_mask(self, trg):
        trg_pad = torch.zeros(trg.shape[0], trg.shape[1], trg.shape[2], device=self.device)

        trg_pad_mask = torch.all(torch.eq(trg, trg_pad), axis=2).unsqueeze(1).unsqueeze(2)

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()

        trg_mask = trg_pad_mask & trg_sub_mask

        return trg_mask

    def forward(self, src, trg1, label):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg1)

        enc_src = self.encoder(src, src_mask)

        output, attention = self.decoder(trg1, enc_src, trg_mask, src_mask)

        """ 
        src_mask_1_2 = self.make_src_mask(output)

        enc_src_1_2 = self.encoder(output, src_mask_1_2)

        trg_mask_1_2 = self.make_trg_mask(enc_src_1_2)

        output_2, attention_2 = self.decoder(src, enc_src_1_2, trg_mask_1_2, src_mask_1_2)
        """
        output_2 = output
        regression_score = self.regression(enc_src)

        return output, output_2, regression_score
        # return output, regression_score

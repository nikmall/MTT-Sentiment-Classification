param = {
    "aligned": 'True',
    "data_path": 'data',
    "batch_size": 32}

param_mctn = {
    "enc_emb_dim": 300,
    "dec_emb_dim": 300,
    "enc_hid_dim": 512,
    "dec_hid_dim": 512,
    "enc_dropout": 0.4,
    "dec_dropout": 0.4,
    "sent_hid_dim": 256,
    "sent_dropout": 0.4,
    "n_layers": 2,
    "n_epochs": 50,
    'loss_dec_weight': 0.1,
    'loss_dec_cycle_weight': 0.1,
    'loss_regress_weight': 1.0
}


param_mtt = {
    "enc_emb_dim": 300,# the len of the vocabulary, in our mosei standart case is the glove300
    "dec_emb_dim": 300,
    "hid_dim": 300,# embedding dimension if glove300 is always 300
    "enc_layers": 3,
    "dec_layers": 3,
    "enc_heads": 3,
    "dec_heads": 3,
    "enc_pf_dim": 360,
    "dec_pf_dim": 360,
    "enc_dropout": 0.31,
    "dec_dropout": 0.31,

    "enc_layers_2": 2,
    "dec_layers_2": 2,
    "enc_heads_2": 3,
    "dec_heads_2": 3,
    "enc_pf_dim_2": 320,
    "dec_pf_dim_2": 100,

    "sent_hid_dim": 156, # 156
    "sent_final_hid": 116, # 132 best
    "sent_dropout": 0.25,
    "sent_n_layers": 1,
    "bidirect": True,

    "transformer_regression": False,

    "n_epochs": 150,
    'lr_patience': 20,
    'loss_dec_weight': 0.1,
    'loss_dec_cycle_weight': 0.1,
    'loss_regress_weight': 0.9,

    'fuse_modalities': False,
    "cyclic": True

}

param_mtt_fuse = {
    "enc_emb_dim": 300,# the len of the vocabulary, in our mosei standart case is the glove300
    "dec_emb_dim": 300,
    "hid_dim": 300,# embedding dimension if glove300 is always 300
    "enc_layers": 3,
    "dec_layers": 3,
    "enc_heads": 6,
    "dec_heads": 6,
    "enc_pf_dim": 490,
    "dec_pf_dim": 490,
    "enc_dropout": 0.31,
    "dec_dropout": 0.31,
    'att_dropout': 0.30,

    "sent_hid_dim": 156,
    "sent_final_hid": 132,
    "sent_dropout": 0.31,
    "sent_n_layers": 2,
    "bidirect": True,

    "transformer_regression": False,

    "n_epochs": 150,
    'lr_patience': 20,
    'loss_dec_weight': 0.15,
    'loss_dec_cycle_weight': 0.10,
    'loss_regress_weight': 0.9,

    'fuse_modalities': True,
    "cyclic": False
}

param_mtt_triple = {
    'hid_dim': 413,
    "output_dim": 301,
    "enc_layers": 3,
    "dec_layers": 3,
    "enc_heads": 7,
    "dec_heads": 7,
    "enc_pf_dim": 590,
    "dec_pf_dim": 490,
    "enc_dropout": 0.31,
    "dec_dropout": 0.31,
    'att_dropout': 0.30,

    "sent_hid_dim": 156,
    "sent_final_hid": 132,
    "sent_dropout": 0.31,
    "sent_n_layers": 2,
    "bidirect": True,

    "transformer_regression": False,

    "n_epochs": 60,
    'lr_patience': 20,
    'loss_dec_weight': 0.10,
    'loss_dec_cycle_weight': 0.10,
    'loss_regress_weight': 0.9,

    'fuse_modalities': True,
    "cyclic": True
}
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
    "enc_emb_dim": 300,
    "dec_emb_dim": 300,
    "hid_dim": 300,
    "enc_layers": 3,
    "dec_layers": 3,
    "enc_heads": 5,
    "dec_heads": 5,
    "enc_pf_dim": 512,
    "dec_pf_dim": 512,
    "enc_dropout": 0.4,
    "dec_dropout": 0.4,
    "max_length": 300,

    "sent_hid_dim": 192,
    "sent_final_hid": 128,
    "sent_dropout": 0.4,
    "sent_n_layers": 2,
    "n_epochs": 50,
    'loss_dec_weight': 0.1,
    'loss_dec_cycle_weight': 0.1,
    'loss_regress_weight': 1.0
}
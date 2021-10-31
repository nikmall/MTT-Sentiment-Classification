
conf = {
  'loss_dec_weight': 0.1,
  'loss_dec_cycle_weight': 0.1,
  'loss_regress_weight': 1.0
}

"""
translation:
  is_bidirectional: True
  hidden_dim: 32
  depth: !!python/tuple [1, 1]  # encoder and decoder depths
  loss_type: 'mse'
  loss_weight: 0.1
  is_cycled: 1
  cycle_loss_type: 'mse'  # only valid if is_cycle
  cycle_loss_weight: 0.1

regression:
  reg_hidden_dim: 32
  l2_factor: 0.01
  loss_type: 'mae'
  loss_weight: 1.0

general:
  train_split: 0.66667  # 2/3
  max_seq_len: 20
  output_dir: './output'
  init_lr: 0.0001
  optim: 'adam'
"""
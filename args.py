import torch

batch_size = 8
num_epochs = 15
vocab_size = 30592  # 30526 + 1
d_model = 768
nhead = 8
n_q_head = 8
n_kv_head = 2
dim_feedforward = 1024
num_layers = 6
learning_rate = 3e-4
max_seq_len = 512
device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_path = r'Squad'
transformer_path = r'transformer_state_dict.pth'
llama_path = r'llama_state_dict.pth'

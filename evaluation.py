import numpy as np
import torch
from datasets import load_from_disk
from tqdm.auto import tqdm
from data import Squad
from model import build_model
from transformers import AutoTokenizer
import evaluate
from args import batch_size, vocab_size, d_model, nhead, n_q_head, n_kv_head, dim_feedforward, num_layers, \
    max_seq_len, device, transformer_path, llama_path
import argparse


def evaluate(args_):
    model_type = args_.model_type
    load = args_.load

    tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token': '[BOS]'})
    tokenizer.add_special_tokens({'eos_token': '[EOS]'})
    tokenizer.add_special_tokens({'additional_special_tokens': ['[CON]', '[QUE]', '[ANS]']})

    data_path = r'Squad'
    dataset = load_from_disk(data_path).train_test_split(test_size=0.1)
    dataset_val = Squad(dataset['test'])

    model = build_model(model_type, d_model, nhead, dim_feedforward, num_layers, vocab_size, max_seq_len, n_q_head,
                        n_kv_head, batch_size)
    print(np.sum([p.numel() for p in model.parameters()]))
    if load:
        if model_type == 'transformer':
            try:
                checkpoint = torch.load(transformer_path)
                model.load_state_dict(checkpoint['model_state_dict'])
            except FileNotFoundError:
                print('No model state dict found. Untrained model will be used.')
        if model_type == 'llama':
            try:
                checkpoint = torch.load(llama_path)
                model.load_state_dict(checkpoint['model_state_dict'])
            except FileNotFoundError:
                print('No model state dict found. Untrained model will be used.')
    model = model.to(device)

    metric_bleu = evaluate.load('sacrebleu')
    metric_bert = evaluate.load('bertscore')

    for batch in tqdm(dataset_val, desc='evaluate validation set'):
        with torch.no_grad():
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("model_type", type=str, help="Name of the model to train")
    parser.add_argument("--load", action='store_true', help="Load previous weights")
    args_ = parser.parse_args()
    evaluate(args_)

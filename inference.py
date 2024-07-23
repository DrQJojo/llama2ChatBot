import argparse
import torch
from transformers import AutoTokenizer
from model import build_model
from args import batch_size, vocab_size, d_model, nhead, n_q_head, n_kv_head, dim_feedforward, num_layers, \
                max_seq_len, transformer_path, llama_path, device


def inference(args_):
    model_type = args_.model_type
    load = True

    tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token': '[BOS]'})
    tokenizer.add_special_tokens({'eos_token': '[EOS]'})
    tokenizer.add_special_tokens({'additional_special_tokens': ['[CON]', '[QUE]', '[ANS]']})

    model = build_model(model_type, d_model, nhead, dim_feedforward, num_layers, vocab_size, max_seq_len, n_q_head,
                        n_kv_head, batch_size)
    if load:
        if model_type == 'transformer':
            try:
                checkpoint = torch.load(transformer_path)
                model.load_state_dict(checkpoint['model_state_dict'])
                print('Transformer state dict loaded.')
            except FileNotFoundError:
                print('No model state dict found. Untrained model will be used.')
        if model_type == 'llama':
            try:
                checkpoint = torch.load(llama_path)
                model.load_state_dict(checkpoint['model_state_dict'])
                print('Llama state dict loaded.')
            except FileNotFoundError:
                print('No model state dict found. Untrained model will be used.')

    model = model.to(device)
    context = "US national day is July 4th."
    question = 'When is US national day?'
    context = 'Sam is my friend.'
    question = 'Who is my friend?'
    print('Context: ', context)
    print('Question: ', question)

    model.eval()
    with torch.no_grad():
        result = model.generate(context, question, tokenizer)
    print(result)

    # context = input('Please input context: ')
    # question = input('Please input question: ')
    # result = model.generate(context, question, tokenizer)
    # print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Use the model to generate output.")
    parser.add_argument("model_type", type=str, help="Name of the model to use")
    args_ = parser.parse_args()
    inference(args_)

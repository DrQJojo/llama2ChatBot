import numpy as np
import matplotlib.pyplot as plt
import time
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW, lr_scheduler
from transformers import AutoTokenizer
from datasets import load_from_disk
from tqdm.auto import tqdm
from data import Squad, Loss, collate_fn
from model import build_model
from args import batch_size, num_epochs, vocab_size, d_model, nhead, n_q_head, n_kv_head, dim_feedforward, num_layers, \
    learning_rate, max_seq_len, device, transformer_path, llama_path
import argparse


def save_checkpoint(model, optimizer, scheduler, epoch, path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, path)


def load_checkpoint(model, optimizer, scheduler, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, scheduler, epoch


def train(args_):
    model_type = args_.model_type
    load = args_.load

    data_path = r'Squad'

    tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token': '[BOS]'})
    tokenizer.add_special_tokens({'eos_token': '[EOS]'})
    tokenizer.add_special_tokens({'additional_special_tokens': ['[CON]', '[QUE]', '[ANS]']})

    dataset = load_from_disk(data_path).train_test_split(test_size=0.1)
    dataset_train = Squad(dataset['train'])
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size,
                                  shuffle=True, collate_fn=collate_fn, num_workers=2)

    model = build_model(model_type, d_model, nhead, dim_feedforward, num_layers, vocab_size, max_seq_len, n_q_head,
                        n_kv_head, batch_size)
    print('Number of parameters in this model: ', np.sum([p.numel() for p in model.parameters()]))
    print('device: ', device)
    model = model.to(device)
    torch.set_float32_matmul_precision('high')
    # model = torch.compile(model)
    optimizer = AdamW(model.parameters(), lr=learning_rate, fused=True)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    # loss_fn = Loss()
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
    start_epoch = 0
    if load:
        if model_type == 'transformer':
            model, optimizer, scheduler, start_epoch = load_checkpoint(model, optimizer, scheduler, path=transformer_path)
        if model_type == 'llama':
            model, optimizer, scheduler, start_epoch = load_checkpoint(model, optimizer, scheduler, path=llama_path)

    lossi = []
    count = 0

    for epoch in tqdm(range(start_epoch, num_epochs), desc='epoch'):
        for batch in tqdm(dataloader_train, desc='training batch'):
            model.train()
            start = time.time()
            input = batch['input'].to(device)
            target = batch['output'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            padding_mask = batch['padding_mask'].to(device)
            ans_index = batch['ans_index']
            with torch.autocast(device_type=device, dtype=torch.float16):
                predict = model(input, attention_mask, padding_mask)
                loss = loss_fn(predict.transpose(1, 2), target)

            optimizer.zero_grad()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            torch.cuda.synchronize()
            end = time.time()
            dt = (end-start)*1000

            if count % 300 == 0:
                print(f'loss: {loss.item():.4f}', f'time: {dt:.2f}ms', f'gradient norm: {norm.item():.3f}')
                lossi.append(loss.item())
                context = "US national day is July 4th."
                question = 'When is US national day?'
                model.eval()
                with torch.no_grad():
                    result = model.generate(context, question, tokenizer)
                print(result)
            count += 1
        if epoch % 3 == 0 and epoch != 0:
            if model_type == 'transformer':
                save_checkpoint(model, optimizer, scheduler, epoch, transformer_path)
            if model_type == 'llama':
                save_checkpoint(model, optimizer, scheduler, epoch, llama_path)

        scheduler.step()

    np.save(r'training_loss.npy', np.array(lossi))
    plt.figure()
    plt.plot(range(len(lossi)), lossi, label='loss', marker='o')
    plt.xlabel('training process')
    plt.ylabel('loss')
    plt.title('loss over training process')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("model_type", type=str, help="Name of the model to train")
    parser.add_argument("--load", action='store_true', help="Load previous weights")
    args_ = parser.parse_args()
    train(args_)

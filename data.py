import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')
tokenizer.add_special_tokens({'bos_token': '[BOS]'})
tokenizer.add_special_tokens({'eos_token': '[EOS]'})
tokenizer.add_special_tokens({'additional_special_tokens': ['[CON]', '[QUE]', '[ANS]']})


class Squad(Dataset):
    def __init__(self, squad):
        super().__init__()
        self.squad = squad

    def __getitem__(self, index):
        item = self.squad[index]
        input = torch.tensor(item['input'])
        output = torch.tensor(item['output'])
        ans_index = torch.tensor(item['ans_index'])
        return {'input': input, 'output': output, 'ans_index': ans_index}

    def __len__(self):
        return len(self.squad)


def generate_mask(sequence):
    seq_len = len(sequence[0])
    padding_mask = (sequence == float(tokenizer.pad_token_id))
    attention_mask = (torch.tril(torch.ones(seq_len, seq_len)) == 0)
    return padding_mask, attention_mask


def collate_fn(batch):
    input = [item['input'] for item in batch]
    output = [item['output'] for item in batch]
    ans_index = [item['ans_index'] for item in batch]

    padded_input_seq = pad_sequence(input, batch_first=True, padding_value=tokenizer.pad_token_id)  # [B,T]
    padding_mask_input, attention_mask_input = generate_mask(padded_input_seq)
    padded_output_seq = pad_sequence(output, batch_first=True, padding_value=tokenizer.pad_token_id)
    return {'input': padded_input_seq, 'output': padded_output_seq, 'padding_mask': padding_mask_input,
            'attention_mask': attention_mask_input, 'ans_index': ans_index}


class Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predict, target, ans_index):
        mask = torch.zeros_like(target)
        for batch_ind, ans_ind in enumerate(ans_index):
            mask[batch_ind, ans_ind:] = 1
        target = target * mask
        loss = torch.nn.CrossEntropyLoss(ignore_index=0)
        return loss(predict.transpose(1, 2), target)

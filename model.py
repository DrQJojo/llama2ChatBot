import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from args import device, max_seq_len


# Vanilla Transformer Decoder
class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, seq_len):
        super().__init__()
        self.word_embd = nn.Embedding(vocab_size, d_model)
        self.pos_embd = torch.zeros((seq_len, d_model), device=device)
        T = torch.arange(0, seq_len, device=device)
        i = torch.arange(0, d_model // 2, device=device)
        div = torch.exp(2 * i / d_model * (-np.log(10000)))
        self.pos_embd[:, ::2] = torch.sin(torch.outer(T, div))
        self.pos_embd[:, 1::2] = torch.cos(torch.outer(T, div))
        self.pos_embd.requires_grad_ = False

    def forward(self, x):
        _, seq_len = x.shape
        embd = self.word_embd(x)
        return embd + self.pos_embd[:seq_len, :]


class Decoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers, vocab_size, max_seq_len):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model, max_seq_len)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                        batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.ffw = nn.Linear(d_model, vocab_size)
        self.max_seq_len = max_seq_len

    def forward(self, x, attention_mask=None, padding_mask=None):
        x = self.embedding(x)
        x = self.decoder(x, x, tgt_mask=attention_mask, tgt_key_padding_mask=padding_mask)
        x = self.ffw(x)
        return x

    def generate(self, context, question, tokenizer):
        sequence = '[CON]' + context + '[QUE]' + question + '[ANS]'
        seq = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sequence))
        new_token = None
        while new_token != tokenizer.eos_token_id:
            if len(seq) > 32:
                break
            input = torch.tensor(seq, device=device).unsqueeze(0)  # [B,T] and B is always 1
            logits = self(input)[:, -1, :]  # [B,C] since we only need the last token during auto-regression
            probs = torch.softmax(logits, dim=-1)
            output = torch.max(probs, dim=-1)[1]  # [B]
            new_token = output.item()
            seq = seq + [new_token]
        return tokenizer.decode(seq)


# Llama
class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x)


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        # x: [B,T,C]
        rms = torch.rsqrt(torch.mean(torch.square(x), dim=-1, keepdim=True) + self.eps)
        return self.alpha * rms * x


class PosEmbedding(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.theta = torch.pow(10000.0, -2 * (torch.arange(1, d_model / 2 + 1) - 1) / d_model)
        self.m = torch.arange(0, seq_len)
        self.matrix = torch.polar(abs=torch.ones_like(torch.outer(self.m, self.theta)),
                                  angle=torch.outer(self.m, self.theta))  # [seq_len,d_model/2]

    def forward(self, x):
        B, T, h, C = x.shape
        x_complex = torch.view_as_complex(x.reshape(B, T, h, -1, 2))  # [B,T,h,C/2]
        matrix_complex = self.matrix.unsqueeze(0).unsqueeze(2).to(x_complex.device)  # [1,seq_len,1,C/2]
        x_out = x_complex * matrix_complex[:, :T, :, :]
        x_out = torch.view_as_real(x_out).reshape(x.shape)
        return x_out


class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.W = nn.Linear(d_model, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, d_model)
        self.V = nn.Linear(d_model, hidden_dim)

    def forward(self, x):
        swish = F.silu(self.W(x))
        x_v = self.V(x)
        return self.W2(swish * x_v)


def repeat_kv(x, n_rep):
    B, T, n_kv_head, head_dim = x.shape
    if n_rep == 1:
        return x
    else:
        return (
            x.unsqueeze(3)
            .expand(B, T, n_kv_head, n_rep,
                    head_dim)  # Returns a new view of the self tensor with singleton dimensions expanded to a larger size.
            .reshape(B, T, n_kv_head * n_rep, head_dim)
        )


class Attention(nn.Module):
    def __init__(self, d_model, n_q_head, n_kv_head, batch_size, seq_len):
        super().__init__()
        self.head_dim = d_model // n_q_head
        self.n_rep = n_q_head // n_kv_head
        self.wq = nn.Linear(d_model, self.head_dim * n_q_head, bias=False)
        self.wk = nn.Linear(d_model, self.head_dim * n_kv_head, bias=False)
        self.wv = nn.Linear(d_model, self.head_dim * n_kv_head, bias=False)
        self.wo = nn.Linear(n_q_head * self.head_dim, d_model, bias=False)

        self.cache_k = torch.zeros(batch_size, seq_len, n_kv_head, self.head_dim, device=device)
        self.cache_v = torch.zeros(batch_size, seq_len, n_kv_head, self.head_dim, device=device)

        self.pos_emb = PosEmbedding(seq_len, self.head_dim)

    def forward(self, x, attention_mask=None, padding_mask=None, start_pos=None):
        B, T, _ = x.shape
        xq = self.wq(x)  # [B,T,head_dim * n_q_head]
        xk = self.wk(x)  # [B,T,head_dim * n_kv_head]
        xv = self.wv(x)  # [B,T,head_dim * n_kv_head]

        xq = xq.reshape(B, T, -1, self.head_dim)
        xk = xk.reshape(B, T, -1, self.head_dim)
        xv = xv.reshape(B, T, -1, self.head_dim)

        # apply rotary position embedding to query and key
        xq = self.pos_emb(xq)
        xk = self.pos_emb(xk)

        # apply kv-cache during inference
        # T is always 1 after prefix
        if not self.training:
            self.cache_k[:B, start_pos:start_pos + T, :, :] = xk
            self.cache_v[:B, start_pos:start_pos + T, :, :] = xv
            xk = self.cache_k[:B, :start_pos + T]
            xv = self.cache_k[:B, :start_pos + T]

        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        xq = xq.transpose(1, 2)  # [B,n_q_head,T,head_dim]
        xk = xk.transpose(1, 2)  # [B,n_q_head,T,head_dim]
        xv = xv.transpose(1, 2)  # [B,n_q_head,T,head_dim]

        # combined_mask = None
        # # attention mask: [T,T]
        # if attention_mask is not None:
        #     attention_mask = attention_mask.unsqueeze(0).unsqueeze(0).logical_not()  # [1,1,T,T]
        #     attention_mask = attention_mask.expand(B, xq.shape[1], T, T)
        #     combined_mask = attention_mask
        # if padding_mask is not None:
        #     padding_mask = padding_mask.unsqueeze(1).unsqueeze(2).logical_not()  # [B,1,1,T]
        #     padding_mask = padding_mask.expand(B, xq.shape[1], T, T)
        #     combined_mask = attention_mask & padding_mask
        # output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=combined_mask)

        attention_score = xq @ xk.transpose(2, 3) / math.sqrt(self.head_dim)  # [B,n_q_head,T,T]
        # attention mask: [T,T]
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)  # [1,1,T,T]
            attention_mask = attention_mask.expand(B, xq.shape[1], T, T)
            attention_score = attention_score.masked_fill(attention_mask == 1, float('-inf'))
        # padding_mask: [B,T]
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,T]
            padding_mask = padding_mask.expand(B, xq.shape[1], T, T)
            attention_score = attention_score.masked_fill(padding_mask == 1, float('-inf'))
        attention_score = F.softmax(attention_score, dim=-1)
        output = attention_score @ xv  # [B,n_q_head,T,head_dim]

        output = output.transpose(1, 2).contiguous().reshape(B, T, -1)
        return self.wo(output)


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_q_head, n_kv_head, batch_size, seq_len, hidden_dim):
        super().__init__()
        self.rms_norm1 = RMSNorm(d_model)
        self.attention = Attention(d_model, n_q_head, n_kv_head, batch_size, seq_len)
        self.rms_norm2 = RMSNorm(d_model)
        self.ffw = FeedForward(d_model, hidden_dim)

    def forward(self, x, attention_mask=None, padding_mask=None, start_pos=None):
        x = x + self.attention(self.rms_norm1(x), attention_mask, padding_mask, start_pos)
        x = x + self.ffw(self.rms_norm1(x))
        return x


class Llama(nn.Module):
    def __init__(self, d_model, n_q_head, n_kv_head, batch_size, seq_len, hidden_dim, vocab_size, n_layers):
        super().__init__()
        self.embd = WordEmbedding(vocab_size, d_model)
        decoder_layer = DecoderBlock(d_model, n_q_head, n_kv_head, batch_size, seq_len, hidden_dim)
        self.decoder = nn.ModuleList([decoder_layer for _ in range(n_layers)])
        self.rms_norm = RMSNorm(d_model)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x, attention_mask=None, padding_mask=None, start_pos=None):
        x = self.embd(x)
        for layer in self.decoder:
            x = layer(x, attention_mask, padding_mask, start_pos)
        x = self.rms_norm(x)
        x = self.linear(x)
        return x

    def generate(self, context, question, tokenizer):
        self.eval()
        sequence = '[CON]' + context + '[QUE]' + question + '[ANS]'
        seq = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sequence))
        ans = seq
        start_pos = len(seq)
        new_token = None
        while new_token != tokenizer.eos_token_id:
            if len(ans) > 32: # max_seq_len:
                break
            input = torch.tensor(seq).unsqueeze(0).to(device)  # [B,T] and B is always 1
            # [B,C] since we only need the last token during auto-regression
            if input.shape[-1] != 1:
                logits = self(input, start_pos=0)[:, -1, :]
            else:
                start_pos += 1
                logits = self(input, start_pos=start_pos)[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            output = torch.max(probs, dim=-1)[1]  # [B]
            new_token = output.item()
            ans = ans + [new_token]
            seq = [new_token]
        return tokenizer.decode(ans)


def build_model(model_type: str, d_model, nhead, dim_feedforward, num_layers, vocab_size, max_seq_len,
                n_q_head, n_kv_head, batch_size):
    assert model_type == 'transformer' or model_type == 'llama', 'Valid model_type: 1.transformer 2.llama'
    if model_type == 'transformer':
        model = Decoder(d_model, nhead, dim_feedforward, num_layers, vocab_size, max_seq_len)
    elif model_type == 'llama':
        model = Llama(d_model, n_q_head, n_kv_head, batch_size, max_seq_len, dim_feedforward, vocab_size, num_layers)
    return model

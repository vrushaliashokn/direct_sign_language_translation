import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple
import torch.nn as nn
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
import numpy as np

class Input(nn.Module):
    def __init__(self, input_dim, num_classes, return_embeddings = False):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        # self.fc = nn.Linear(input_dim, num_classes)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x, return_embeddings=False):
        x = self.fc1(x)
        x = self.relu(x)   # ← this is your embedding
        outputs = self.fc2(x)
        return x, outputs

# Source - https://stackoverflow.com/a/77445896
# Posted by Yakov Dan, modified by community. See post 'Timeline' for change history
# Retrieved 2026-04-02, License - CC BY-SA 4.0

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout = 0.1, max_len = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, T, D)
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, input_dim, ntoken, d_model, nhead, d_hid, nlayers, dropout = 0.1, batch_first=False, norm_first=True):
        super().__init__()
        self.model_type = 'Transformer'
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, ntoken)
        self.norm = nn.LayerNorm(d_model)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask=None):
        mask = (src.abs().sum(dim=-1) == 0)  # (B, T) — compute BEFORE projectiomn
        src = self.input_proj(src)
        src = src * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        src = src.transpose(0, 1)
        enc_output = self.transformer_encoder(src, src_key_padding_mask=mask)
        enc_output = enc_output.transpose(0, 1)
        enc_output = self.norm(enc_output)
        output = enc_output.mean(dim=1)
        final = self.linear(output)
        return enc_output, output, final

class ASLTransformerModel(nn.Module):
    def __init__(self, input_dim, ntoken, d_model, nhead, d_hid, nlayers, device, dropout = 0.1, batch_first=False, norm_first=True):
        super().__init__()
        self.isl_model = TransformerModel(input_dim, ntoken=220, d_model = 64, nhead=4, d_hid=128, nlayers=2, dropout = 0.5, batch_first=True, norm_first=True).to(device)
        self.isl_model.load_state_dict(torch.load(f'isl_transformer_BACKBONE.pt'))
        for param in self.isl_model.parameters():
            param.requires_grad = True
        self.model_type = 'Transformer'
        self.temporal_conv = nn.Conv1d(
        in_channels=d_model,
        out_channels=d_model,
        kernel_size=3,
        padding=1
        )
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, ntoken)
        
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask=None):
        mask = (src.abs().sum(dim=-1) == 0)  # (B, T) — compute BEFORE projection
        enc_output, _, _ = self.isl_model(src)
        src = enc_output.transpose(1, 2)
        src = self.temporal_conv(src)
        src = src.transpose(1, 2)
        src = src * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        src = src.transpose(0, 1)
        asl_enc_output = self.transformer_encoder(src, src_key_padding_mask=mask)
        output = asl_enc_output.transpose(0, 1)
        output = output.mean(dim=1)
        final = self.linear(output)
        return output, final
        
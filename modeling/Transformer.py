
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError("activation should be relu/gelu, not {}".format(activation))


class TransformerEncoderDownstream(nn.Module):

    def __init__(self, input_dimension, output_dimension, hidden_dimmension,
                 attention_heads, encoder_number_of_layers,  
                 positional_encodings, dropout, dim_feedforward=512, activation='gelu'):
        
        super(TransformerEncoderDownstream, self).__init__()

        self.project_input = nn.Linear(input_dimension, hidden_dimmension)

        self.hidden_dimmension = hidden_dimmension
        if attention_heads is None:
            attention_heads=hidden_dimmension//64
        self.attention_heads = attention_heads
        self.positional_encodings = positional_encodings

        self.encoder = nn.Linear(input_dimension, hidden_dimmension) # using linear projection instead
        self.pos_encoder = PositionalEncoding(hidden_dimmension, dropout)
        print(hidden_dimmension, attention_heads)
        encoder_layer = TransformerEncoderLayer(hidden_dimmension, self.attention_heads, dim_feedforward, dropout, activation=activation)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, encoder_number_of_layers)

        self.output_layer = nn.Linear(450*2, output_dimension)

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        
        self.sigmoid = nn.Sigmoid()
        

        self.conv1d = nn.Conv1d(in_channels=hidden_dimmension,out_channels=2,kernel_size=128)

    def forward(self, src):#, padding_masks):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, seq_length, feat_dim)
        """

        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        src = self.project_input(src)                                                     # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        if self.positional_encodings:                                                   # add positional encoding
            src = self.pos_encoder(src)
                                                         
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        # output = self.transformer_encoder(src, src_key_padding_mask=~padding_masks)     # (seq_length, batch_size, d_model)
        output = self.transformer_encoder(src)     # (seq_length, batch_size, d_model)
        output = self.act(output)                                                       # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout1(output)
        # Most probably defining a Linear(d_model,feat_dim) vectorizes the operation over (seq_length, batch_size).
        
        # output at this point is of shape (450, batch_size, hidden_dim)
        output = output.permute(1,2,0) # convert shape into [batch, hidden_dim, 450] (or N,C,L for a conv1d)
        # convolving
        output = self.conv1d(output)
        # padding
        padding_size = 450-output.shape[-1]
        paddings = (int(np.floor(padding_size/2)),int(np.ceil(padding_size/2)))
        output = F.pad(output, paddings) # this would return a tensor of shape (batch, 2,450)

        # reshape and Linear
        output = torch.reshape(output, (output.shape[0], output.shape[1]*output.shape[2]))
        output = self.output_layer(output)  # (batch_size, seq_length, feat_dim)
        output = self.sigmoid(output)

        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=501):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.scale * self.pe[:x.size(0), :]
        return self.dropout(x)
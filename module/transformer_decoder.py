import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _single
#import module.utils as utils
from module.multihead_attention import MultiheadAttention
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import copy

def make_positions(tensor, padding_idx, left_pad):
    """Replace non-padding symbols with their position numbers.
    Position numbers begin at padding_idx+1.
    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """

    # creates tensor from scratch - to avoid multigpu issues
    max_pos = padding_idx + 1 + tensor.size(1)
    #if not hasattr(make_positions, 'range_buf'):
    range_buf = tensor.new()
    #make_positions.range_buf = make_positions.range_buf.type_as(tensor)
    if range_buf.numel() < max_pos:
        torch.arange(padding_idx + 1, max_pos, out=range_buf)
    mask = tensor.ne(padding_idx)
    positions = range_buf[:tensor.size(1)].expand_as(tensor)
    if left_pad:
        positions = positions - mask.size(1) + mask.long().sum(dim=1).unsqueeze(1)

    out = tensor.clone()
    out = out.masked_scatter_(mask,positions[mask])
    return out

class TransformerDecoderLayer(nn.Module):
    """Decoder layer block."""

    def __init__(self, embed_dim, n_att, dropout=0.5, normalize_before=True, last_ln=False):
        super().__init__()

        self.embed_dim = embed_dim
        self.dropout = dropout
        self.relu_dropout = dropout
        self.normalize_before = normalize_before
        num_layer_norm = 3

        # self-attention on generated recipe
        self.self_attn = MultiheadAttention(
            self.embed_dim, n_att,
            dropout=dropout,
        )

        self.cond_att = MultiheadAttention(
            self.embed_dim, n_att,
            dropout=dropout,
        )

        self.fc1 = Linear(self.embed_dim, self.embed_dim)
        self.fc2 = Linear(self.embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for i in range(num_layer_norm)])
        self.use_last_ln = last_ln
        if self.use_last_ln:
            self.last_ln = LayerNorm(self.embed_dim)

    def forward(self, x, incremental_state, img_features):

        # self attention
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            mask_future_timesteps=True,
            incremental_state=incremental_state,
            need_weights=False,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)

        # attention
        x, _ = self.cond_att(query=x,
                             key=img_features,
                             value=img_features,
                             key_padding_mask=None,
                             incremental_state=incremental_state,
                             static_kv=True,
                                    )

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)

        residual = x
        x = self.maybe_layer_norm(-1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(-1, x, after=True)

        if self.use_last_ln:
            x = self.last_ln(x)

        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x

class DecoderTransformer(nn.Module):
    """Transformer decoder."""

    def __init__(self, embed_size, vocab_size, dropout=0.5, seq_length=20, num_instrs=15,
                 attention_nheads=16, pos_embeddings=True, num_layers=8, learned=True, normalize_before=True,
                 normalize_inputs=False, last_ln=False, scale_embed_grad=False):
        super(DecoderTransformer, self).__init__()
        self.dropout = dropout
        self.seq_length = seq_length * num_instrs
        self.embed_tokens = nn.Embedding(vocab_size, embed_size, padding_idx=vocab_size-1,
                                         scale_grad_by_freq=scale_embed_grad)
        nn.init.normal_(self.embed_tokens.weight, mean=0, std=embed_size ** -0.5)
        if pos_embeddings:
            self.embed_positions = PositionalEmbedding(1024, embed_size, 0, left_pad=False, learned=learned)
        else:
            self.embed_positions = None
        self.normalize_inputs = normalize_inputs
        if self.normalize_inputs:
            self.layer_norms_in = nn.ModuleList([LayerNorm(embed_size) for i in range(3)])

        self.embed_scale = math.sqrt(embed_size)
        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderLayer(embed_size, attention_nheads, dropout=dropout, normalize_before=normalize_before,
                                    last_ln=last_ln)
            for i in range(num_layers)
        ])

        self.linear = Linear(embed_size, vocab_size-1)

    def forward(self, captions, img_features, incremental_state=None):

        if img_features is not None:
            img_features = img_features.permute(0, 2, 1)
            img_features = img_features.transpose(0, 1)
            if self.normalize_inputs:
                self.layer_norms_in[1](img_features)

        # embed positions
        if self.embed_positions is not None:
            positions = self.embed_positions(captions, incremental_state=incremental_state)
        if incremental_state is not None:
            if self.embed_positions is not None:
                positions = positions[:, -1:]
            captions = captions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(captions)

        if self.embed_positions is not None:
            x += positions

        if self.normalize_inputs:
            x = self.layer_norms_in[2](x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        for p, layer in enumerate(self.layers):
            x  = layer(
                x,
                incremental_state,
                img_features
            )

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        x = self.linear(x)
        _, predicted = x.max(dim=-1)

        return x, predicted

def PositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad, learned=False):
    if learned:
        m = LearnedPositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad)
        nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
        nn.init.constant_(m.weight[padding_idx], 0)
    else:
        m = SinusoidalPositionalEmbedding(embedding_dim, padding_idx, left_pad, num_embeddings)
    return m

class LearnedPositionalEmbedding(nn.Embedding):
    """This module learns positional embeddings up to a fixed maximum size.
    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """

    def __init__(self, num_embeddings, embedding_dim, padding_idx, left_pad):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.left_pad = left_pad
        nn.init.normal_(self.weight, mean=0, std=embedding_dim ** -0.5)

    def forward(self, input, incremental_state=None):
        """Input is expected to be of size [bsz x seqlen]."""
        if incremental_state is not None:
            # positions is the same for every token when decoding a single step

            positions = input.data.new(1, 1).fill_(self.padding_idx + input.size(1))
        else:

            positions = make_positions(input.data, self.padding_idx, self.left_pad)
        return super().forward(positions)

    def max_positions(self):
        """Maximum number of supported positions."""
        return self.num_embeddings - self.padding_idx - 1

class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.
    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """

    def __init__(self, embedding_dim, padding_idx, left_pad, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.left_pad = left_pad
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )
        self.register_buffer('_float_tensor', torch.FloatTensor())

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input, incremental_state=None):
        """Input is expected to be of size [bsz x seqlen]."""
        # recompute/expand embeddings if needed
        bsz, seq_len = input.size()
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.type_as(self._float_tensor)

        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            return self.weights[self.padding_idx + seq_len, :].expand(bsz, 1, -1)

        positions = make_positions(input.data, self.padding_idx, self.left_pad)
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    nn.init.constant_(m.bias, 0.)
    return m

def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim)
    return m

def Embedding(num_embeddings, embedding_dim, padding_idx, ):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    return m

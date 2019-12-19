import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _single
#import module.utils as utils
#from module.multihead_attention import MultiheadAttention
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import copy

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

    def forward(self, ingr_features, ingr_mask, captions, img_features, incremental_state=None):

        if ingr_features is not None:
            ingr_features = ingr_features.permute(0, 2, 1)
            ingr_features = ingr_features.transpose(0, 1)
            if self.normalize_inputs:
                self.layer_norms_in[0](ingr_features)

        if img_features is not None:
            img_features = img_features.permute(0, 2, 1)
            img_features = img_features.transpose(0, 1)
            if self.normalize_inputs:
                self.layer_norms_in[1](img_features)

        if ingr_mask is not None:
            ingr_mask = (1-ingr_mask.squeeze(1)).byte()

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
                ingr_features,
                ingr_mask,
                incremental_state,
                img_features
            )
            
        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        x = self.linear(x)
        _, predicted = x.max(dim=-1)

        return x, predicted
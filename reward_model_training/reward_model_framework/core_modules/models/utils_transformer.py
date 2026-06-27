import numpy as np
import torch
import torch.nn as nn


def create_fourier_features(num_positions, num_features, embed_dim):
    position = torch.arange(num_positions, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, num_features, 2).float() * (-np.log(10000.0) / num_features))
    pos_features = torch.zeros((num_positions, embed_dim))
    pos_features[:, : num_features // 2] = torch.sin(position * div_term)
    pos_features[:, num_features // 2 : num_features] = torch.cos(position * div_term)
    return pos_features


class SequenceEncoderWithFourier(nn.Module):
    def __init__(self, input_dim=3, embed_dim=128, num_heads=4, forward_expansion=4, dropout=0.1, max_length=98, num_features=128):
        super().__init__()
        self.embed = nn.Linear(input_dim, embed_dim)
        self.pos_embedding = create_fourier_features(max_length, num_features, embed_dim)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * forward_expansion, dropout=dropout, activation="gelu")
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=4)

        self.stn = None  # Set externally if needed
        self.conv1 = None
        self.conv2 = None
        self.conv3 = None
        self.bn1 = None
        self.bn2 = None
        self.bn3 = None

    def forward(self, x):
        seq_length = x.size(1)
        if self.stn is not None:
            x_stn = x.transpose(2, 1)
            trans = self.stn(x_stn)
            x = torch.bmm(x_stn.transpose(2, 1), trans).transpose(2, 1)
        if self.conv1 is not None:
            x = torch.nn.functional.softplus(self.bn1(self.conv1(x)))
            x = torch.nn.functional.softplus(self.bn2(self.conv2(x)))
            x = torch.nn.functional.softplus(self.bn3(self.conv3(x)))
            x = x.transpose(2, 1)
        x = self.embed(x) + self.pos_embedding[:seq_length, :].to(x.device)
        x = self.transformer_encoder(x)
        return x


class SequenceEncoderWithFourierSigma(nn.Module):
    def __init__(self, input_dim=3, embed_dim=128, num_heads=4, forward_expansion=4, dropout=0.1, max_length=98, num_features=128):
        super().__init__()
        self.embed = nn.Linear(input_dim, embed_dim)
        self.pos_embedding = create_fourier_features(max_length, num_features, embed_dim)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * forward_expansion, dropout=dropout, activation="gelu")
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=4)

    def forward(self, x):
        seq_length = x.size(1)
        x = x + self.pos_embedding[:seq_length, :].to(x.device)
        x = self.transformer_encoder(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query):
        attention = self.attention(query, key, value)[0]
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class TransformerGPT(nn.Module):
    def __init__(self, embed_size=256, heads=4, depth=4, forward_expansion=4, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out, out, out)
        return out


class CrossAttentionModule(nn.Module):
    def __init__(self, embed_dim, num_heads, cross_attention_dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=cross_attention_dropout, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, query, key, value, attn_mask=None, keyscale=None):
        query_norm = self.layer_norm1(query)
        key_norm = self.layer_norm1(key)
        value_norm = self.layer_norm1(value)
        if keyscale is not None:
            key_norm = key_norm * keyscale
        attn_output, _ = self.attn(query_norm, key_norm, value_norm, attn_mask=attn_mask)
        attn_output = self.layer_norm2(attn_output + query)
        return attn_output

    def get_attention_maps(self, query, key, value, attn_mask=None, keyscale=None):
        query_norm = self.layer_norm1(query)
        key_norm = self.layer_norm1(key)
        value_norm = self.layer_norm1(value)
        if keyscale is not None:
            key_norm = key_norm * keyscale
        _, attn_map = self.attn(query_norm, key_norm, value_norm, average_attn_weights=False, attn_mask=attn_mask)
        return attn_map


__all__ = [
    "create_fourier_features",
    "SequenceEncoderWithFourier",
    "SequenceEncoderWithFourierSigma",
    "TransformerBlock",
    "TransformerGPT",
    "CrossAttentionModule",
]

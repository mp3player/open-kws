"""Inference-only GRU and noncausal CNN encoders matching released checkpoints."""
import hashlib

import torch
from torch import nn
from torch.nn import functional


def encoder_fingerprint(encoder):
    digest = hashlib.sha256()
    if encoder.encoder_type == "cnn":
        prefix = b"content_encoder:noncausal_cnn:relu:v1\0" if encoder.config.get("cnn_activation") == "relu" else b"content_encoder:noncausal_cnn:v1\0"
        digest.update(prefix)
    for name, tensor in sorted(encoder.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class TemporalLayer(nn.Module):
    def __init__(self, inputs, outputs, kernel_size, stride=1, dilation=1, dropout=.1, activation="gelu"):
        super().__init__()
        total_padding = (kernel_size - 1) * dilation
        self.padding = (total_padding // 2, total_padding - total_padding // 2)
        self.stride = stride
        self.convolution = nn.Conv1d(inputs, outputs, kernel_size, stride=stride, dilation=dilation)
        self.normalization = nn.LayerNorm(outputs)
        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, lengths):
        features = self.convolution(functional.pad(features.transpose(1, 2), self.padding)).transpose(1, 2)
        lengths = (lengths + self.stride - 1) // self.stride
        features = self.dropout(self.activation(self.normalization(features)))
        mask = torch.arange(features.size(1), device=features.device)[None, :] < lengths[:, None]
        return features.masked_fill(~mask.unsqueeze(-1), 0), lengths


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels, dilation, dropout, activation="gelu"):
        super().__init__()
        self.first = TemporalLayer(channels, channels, 3, dilation=dilation, dropout=dropout, activation=activation)
        self.second = TemporalLayer(channels, channels, 3, dilation=dilation, dropout=dropout, activation=activation)

    def forward(self, features, lengths):
        residual, _ = self.first(features, lengths)
        residual, _ = self.second(residual, lengths)
        return features + residual, lengths


class ContentEncoder(nn.Module):
    def __init__(self, n_mels=40, projection_size=64, embedding_dim=128, dropout=.1,
                 encoder_type="gru", cnn_channels=256, cnn_blocks=6, cnn_activation="gelu", hidden_size=96, layers=2):
        super().__init__()
        if encoder_type not in {"cnn", "gru"} or cnn_activation not in {"gelu", "relu"}:
            raise ValueError("This inference package supports GRU and noncausal CNN encoders")
        if encoder_type == "gru" and cnn_activation != "gelu":
            raise ValueError("CNN activation only applies to CNN encoders")
        if min(n_mels, projection_size, embedding_dim, cnn_blocks, hidden_size, layers) < 1 or cnn_channels < 2 or not 0 <= dropout < 1:
            raise ValueError("Invalid encoder dimensions")
        self.encoder_type = encoder_type
        self.config = dict(n_mels=n_mels, projection_size=projection_size, embedding_dim=embedding_dim,
                           dropout=dropout, encoder_type=encoder_type, cnn_channels=cnn_channels, cnn_blocks=cnn_blocks)
        if cnn_activation != "gelu":
            self.config["cnn_activation"] = cnn_activation
        activation = nn.GELU() if cnn_activation == "gelu" else nn.ReLU()
        self.projection = nn.Sequential(nn.Linear(n_mels, projection_size), nn.LayerNorm(projection_size), activation)
        if encoder_type == "gru":
            self.config = dict(n_mels=n_mels, projection_size=projection_size, hidden_size=hidden_size,
                               layers=layers, embedding_dim=embedding_dim, dropout=dropout)
            self.recurrent = nn.GRU(projection_size, hidden_size, layers, batch_first=True,
                                    dropout=dropout if layers > 1 else 0., bidirectional=False)
            feature_size = hidden_size
        else:
            self.temporal = nn.ModuleList([
                TemporalLayer(projection_size, cnn_channels // 2, 5, stride=2, dropout=dropout, activation=cnn_activation),
                TemporalLayer(cnn_channels // 2, cnn_channels, 5, stride=2, dropout=dropout, activation=cnn_activation),
                *(TemporalResidualBlock(cnn_channels, 2 ** position, dropout, cnn_activation) for position in range(cnn_blocks)),
            ])
            feature_size = cnn_channels
        self.attention = nn.Sequential(nn.Linear(feature_size, 64), nn.Tanh(), nn.Linear(64, 1))
        self.embedding = nn.Sequential(nn.Linear(feature_size * 2, embedding_dim), nn.LayerNorm(embedding_dim))

    def forward_features(self, inputs, lengths):
        lengths = lengths.detach().cpu().long()
        if lengths.shape != (len(inputs),) or torch.any(lengths < 1) or torch.any(lengths > inputs.size(1)):
            raise ValueError("Invalid utterance lengths")
        projected = self.projection(inputs)
        if self.encoder_type == "gru":
            packed = nn.utils.rnn.pack_padded_sequence(projected, lengths, batch_first=True, enforce_sorted=False)
            packed, _ = self.recurrent(packed)
            features, _ = nn.utils.rnn.pad_packed_sequence(packed, batch_first=True, total_length=inputs.size(1))
            return features, lengths.to(inputs.device)
        lengths = lengths.to(inputs.device)
        mask = torch.arange(inputs.size(1), device=inputs.device)[None, :] < lengths[:, None]
        features = projected.masked_fill(~mask.unsqueeze(-1), 0)
        for layer in self.temporal:
            features, lengths = layer(features, lengths)
        return features, lengths

    def forward(self, inputs, lengths):
        features, lengths = self.forward_features(inputs, lengths)
        mask = torch.arange(features.size(1), device=inputs.device)[None, :] < lengths[:, None]
        weights = self.attention(features).squeeze(-1).float().masked_fill(~mask, float("-inf")).softmax(-1)
        pooled = (features * weights.unsqueeze(-1)).sum(1)
        last = features[torch.arange(len(inputs), device=inputs.device), lengths - 1]
        return self.embedding(torch.cat((pooled, last), dim=-1))

    def embed(self, inputs, lengths):
        return functional.normalize(self(inputs, lengths).float(), dim=-1)

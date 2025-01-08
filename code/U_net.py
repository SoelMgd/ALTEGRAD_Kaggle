import torch
import torch.nn as nn
import torch.nn.functional as F

# Attention Block for Cross-Attention with Statistics
class CrossAttentionBlock(nn.Module):
    def __init__(self, latent_dim, cond_dim):
        super(CrossAttentionBlock, self).__init__()
        self.q_proj = nn.Linear(latent_dim, latent_dim)
        self.k_proj = nn.Linear(cond_dim, latent_dim)
        self.v_proj = nn.Linear(cond_dim, latent_dim)
        self.out_proj = nn.Linear(latent_dim, latent_dim)

    def forward(self, x, cond):
        # Query, Key, Value
        Q = self.q_proj(x)  # [batch_size, latent_dim]
        K = self.k_proj(cond)  # [batch_size, cond_dim]
        V = self.v_proj(cond)  # [batch_size, cond_dim]

        # Attention
        attn_weights = torch.softmax(Q @ K.T / (K.size(-1) ** 0.5), dim=-1)
        attn_output = attn_weights @ V

        return self.out_proj(attn_output) + x


# U-Net Block
class UNetBlock(nn.Module):
    def __init__(self, latent_dim, cond_dim):
        super(UNetBlock, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.BatchNorm1d(latent_dim),
        )
        self.cross_attention = CrossAttentionBlock(latent_dim, cond_dim)
        self.layer2 = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.BatchNorm1d(latent_dim),
        )

    def forward(self, x, cond):
        x = self.layer1(x)
        x = self.cross_attention(x, cond)
        x = self.layer2(x)
        return x


# U-Net with Cross-Attention
class UNet(nn.Module):
    def __init__(self, latent_dim, cond_dim, n_layers):
        super(UNet, self).__init__()
        self.encoder = nn.ModuleList([UNetBlock(latent_dim, cond_dim) for _ in range(n_layers)])
        self.middle = UNetBlock(latent_dim, cond_dim)
        self.decoder = nn.ModuleList([UNetBlock(latent_dim, cond_dim) for _ in range(n_layers)])

    def forward(self, x, cond):
        # Encoder path
        skip_connections = []
        for block in self.encoder:
            x = block(x, cond)
            skip_connections.append(x)

        # Middle block
        x = self.middle(x, cond)

        # Decoder path with skip connections
        for block, skip in zip(self.decoder, reversed(skip_connections)):
            x = block(x + skip, cond)

        return x

import torch
import math
import torch.nn as nn
import torch.nn.functional as F


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


# forward diffusion (using the nice property)
def q_sample(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


# Loss function for denoising
def p_losses(denoise_model, x_start, t, cond, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None, loss_type="l1"):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=noise)
    predicted_noise = denoise_model(x_noisy, t, cond)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss


# Position embeddings
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

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
    


@torch.no_grad()
def p_sample(model, x, t, cond, t_index, betas):
    # define alphas
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    # calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t, cond) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise

# Algorithm 2 (including returning all images)
@torch.no_grad()
def p_sample_loop(model, cond, timesteps, betas, shape):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []

    for i in reversed(range(0, timesteps)):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), cond, i, betas)
        imgs.append(img)
        #imgs.append(img.cpu().numpy())
    return imgs



@torch.no_grad()
def sample(model, cond, latent_dim, timesteps, betas, batch_size):
    return p_sample_loop(model, cond, timesteps, betas, shape=(batch_size, latent_dim))

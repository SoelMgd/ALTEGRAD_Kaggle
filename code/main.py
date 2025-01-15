import argparse
import os
import random
import scipy as sp
import pickle

import shutil
import csv
import ast

import scipy.sparse as sparse
from tqdm import tqdm
from torch import Tensor
import networkx as nx
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
from torch_geometric.data import Data

import torch.nn.functional as F
from torch_geometric.loader import DataLoader

# === PLOTTING ADDITIONS START ===
import matplotlib
matplotlib.use('Agg')  # Comment out if using Jupyter notebooks
import matplotlib.pyplot as plt
# === PLOTTING ADDITIONS END ===

from autoencoder import VariationalAutoEncoder
from denoise_model import DenoiseNN, p_losses, sample
from utils import linear_beta_schedule, construct_nx_from_adj, preprocess_dataset
from MAE import compute_mae

from torch.utils.data import Subset
np.random.seed(13)

# Argument parser
parser = argparse.ArgumentParser(description='NeuralGraphGenerator')

# Learning rate for the optimizer
parser.add_argument('--lr', type=float, default=1e-3)

# Dropout rate
parser.add_argument('--dropout', type=float, default=0.5)

# Batch size for training
parser.add_argument('--batch-size', type=int, default=2*256)

# Number of epochs for the autoencoder training
parser.add_argument('--epochs-autoencoder', type=int, default=200)

# Hidden dimension size for the encoder network
parser.add_argument('--hidden-dim-encoder', type=int, default=64)

# Hidden dimension size for the decoder network
parser.add_argument('--hidden-dim-decoder', type=int, default=256)

# Dimensionality of the latent space
parser.add_argument('--latent-dim', type=int, default=32)

# Maximum number of nodes of graphs
parser.add_argument('--n-max-nodes', type=int, default=50)

# Number of layers in the encoder network
parser.add_argument('--n-layers-encoder', type=int, default=2)

# Number of layers in the decoder network
parser.add_argument('--n-layers-decoder', type=int, default=3)

# Dimensionality of spectral embeddings
parser.add_argument('--spectral-emb-dim', type=int, default=10)

# Number of training epochs for the denoising model
parser.add_argument('--epochs-denoise', type=int, default=100)

# Number of timesteps in the diffusion
parser.add_argument('--timesteps', type=int, default=500)

# Hidden dimension size for the denoising model
parser.add_argument('--hidden-dim-denoise', type=int, default=512)

# Number of layers in the denoising model
parser.add_argument('--n-layers_denoise', type=int, default=3)

# Flags for training steps
parser.add_argument('--train-autoencoder', action='store_false', default=True)
parser.add_argument('--train-denoiser', action='store_true', default=True)

# Conditioning vector dimension
parser.add_argument('--dim-condition', type=int, default=128)
parser.add_argument('--n-condition', type=int, default=7)

# Contrastive hyperparams (if you want them easily tunable)
parser.add_argument('--beta-kld', type=float, default=0.05, help='Weight for KLD term')
parser.add_argument('--alpha-cont', type=float, default=0.1, help='Weight for contrastive term')
parser.add_argument('--temp-cont', type=float, default=0.07, help='Temperature for contrastive loss')

args = parser.parse_args()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# preprocess datasets
trainset = preprocess_dataset("train", args.n_max_nodes, args.spectral_emb_dim)
validset = preprocess_dataset("valid", args.n_max_nodes, args.spectral_emb_dim)
testset = preprocess_dataset("test", args.n_max_nodes, args.spectral_emb_dim)

train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(validset, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

# initialize VGAE model (with contrastive-ready code from autoencoder.py)
autoencoder = VariationalAutoEncoder(
    args.spectral_emb_dim+1, 
    args.hidden_dim_encoder, 
    args.hidden_dim_decoder, 
    args.latent_dim, 
    args.n_layers_encoder, 
    args.n_layers_decoder, 
    args.n_max_nodes
).to(device)

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

# Generate a time-stamped suffix for plot filenames to avoid overwriting
time_suffix = datetime.now().strftime('%d%m%Y_%H%M')

# === PLOTTING ADDITIONS START ===
# Lists to track autoencoder losses over epochs
train_loss_list_ae = []
train_recon_list_ae = []
train_kld_list_ae = []
train_cont_list_ae = []

val_loss_list_ae = []
val_recon_list_ae = []
val_kld_list_ae = []
val_cont_list_ae = []
# === PLOTTING ADDITIONS END ===

# Train VGAE model
if args.train_autoencoder:
    best_val_loss = np.inf

    for epoch in range(1, args.epochs_autoencoder + 1):
        autoencoder.train()

        train_loss_all = 0.0
        train_recon_all = 0.0
        train_kld_all   = 0.0
        train_cont_all  = 0.0
        cnt_train = 0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            # Now we call loss_function which returns (total_loss, recon, kld, cont)
            total_loss, recon_loss, kld_loss, cont_loss = autoencoder.loss_function(
                data,
                beta=args.beta_kld,
                alpha=args.alpha_cont,
                temperature=args.temp_cont
            )
            total_loss.backward()
            optimizer.step()

            train_loss_all   += total_loss.item()
            train_recon_all  += recon_loss.item()
            train_kld_all    += kld_loss.item()
            train_cont_all   += cont_loss.item()
            cnt_train        += 1

        mean_train_loss  = train_loss_all  / cnt_train
        mean_train_recon = train_recon_all / cnt_train
        mean_train_kld   = train_kld_all   / cnt_train
        mean_train_cont  = train_cont_all  / cnt_train

        # Validation
        autoencoder.eval()
        val_loss_all   = 0.0
        val_recon_all  = 0.0
        val_kld_all    = 0.0
        val_cont_all   = 0.0
        cnt_val        = 0

        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                total_loss, recon_loss, kld_loss, cont_loss = autoencoder.loss_function(
                    data,
                    beta=args.beta_kld,
                    alpha=args.alpha_cont,
                    temperature=args.temp_cont
                )
                val_loss_all   += total_loss.item()
                val_recon_all  += recon_loss.item()
                val_kld_all    += kld_loss.item()
                val_cont_all   += cont_loss.item()
                cnt_val        += 1

        mean_val_loss  = val_loss_all  / cnt_val
        mean_val_recon = val_recon_all / cnt_val
        mean_val_kld   = val_kld_all   / cnt_val
        mean_val_cont  = val_cont_all  / cnt_val

        # === Store for plotting
        train_loss_list_ae.append(mean_train_loss)
        train_recon_list_ae.append(mean_train_recon)
        train_kld_list_ae.append(mean_train_kld)
        train_cont_list_ae.append(mean_train_cont)

        val_loss_list_ae.append(mean_val_loss)
        val_recon_list_ae.append(mean_val_recon)
        val_kld_list_ae.append(mean_val_kld)
        val_cont_list_ae.append(mean_val_cont)

        dt_t = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        print(
            f'{dt_t} Epoch: {epoch:04d}, '
            f'Train Loss: {mean_train_loss:.5f}, Recon: {mean_train_recon:.2f}, '
            f'KLD: {mean_train_kld:.2f}, Cont: {mean_train_cont:.2f}, '
            f'Val Loss: {mean_val_loss:.5f}, Val Recon: {mean_val_recon:.2f}, '
            f'Val KLD: {mean_val_kld:.2f}, Val Cont: {mean_val_cont:.2f}'
        )

        scheduler.step()

        # Save best
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            torch.save({
                'state_dict': autoencoder.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, 'autoencoder.pth.tar')

    # === Plotting at the end of autoencoder training
    epochs_range = range(1, args.epochs_autoencoder + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, train_loss_list_ae, label='Train Loss')
    plt.plot(epochs_range, val_loss_list_ae, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('VGAE: Total Loss')
    plt.legend()
    plt.savefig(f'Training/vgae_total_loss_{time_suffix}.png')
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, train_recon_list_ae, label='Train Recon')
    plt.plot(epochs_range, val_recon_list_ae, label='Val Recon')
    plt.xlabel('Epoch')
    plt.ylabel('Reconstruction Loss')
    plt.title('VGAE: Reconstruction Loss')
    plt.legend()
    plt.savefig(f'Training/vgae_recon_loss_{time_suffix}.png')
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, train_kld_list_ae, label='Train KLD')
    plt.plot(epochs_range, val_kld_list_ae, label='Val KLD')
    plt.xlabel('Epoch')
    plt.ylabel('KLD Loss')
    plt.title('VGAE: KLD Loss')
    plt.legend()
    plt.savefig(f'Training/vgae_kld_loss_{time_suffix}.png')
    plt.close()

    # Contrastive loss plot
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, train_cont_list_ae, label='Train Contrastive')
    plt.plot(epochs_range, val_cont_list_ae, label='Val Contrastive')
    plt.xlabel('Epoch')
    plt.ylabel('Contrastive Loss')
    plt.title('VGAE: Contrastive Loss')
    plt.legend()
    plt.savefig(f'Training/vgae_contrastive_loss_{time_suffix}.png')
    plt.close()

else:
    checkpoint = torch.load('autoencoder.pth.tar')
    autoencoder.load_state_dict(checkpoint['state_dict'])

autoencoder.eval()

# =======================
#     DIFFUSION PART
# =======================
betas = linear_beta_schedule(timesteps=args.timesteps)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

# initialize denoising model
denoise_model = DenoiseNN(
    input_dim=args.latent_dim, 
    hidden_dim=args.hidden_dim_denoise, 
    n_layers=args.n_layers_denoise, 
    n_cond=args.n_condition, 
    d_cond=args.dim_condition
).to(device)

optimizer = torch.optim.Adam(denoise_model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

# === PLOTTING ADDITIONS START ===
train_loss_list_dn = []
val_loss_list_dn = []
# === PLOTTING ADDITIONS END ===

if args.train_denoiser:
    best_val_loss = np.inf
    for epoch in range(1, args.epochs_denoise+1):
        denoise_model.train()
        train_loss_all = 0.0
        train_count = 0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            x_g = autoencoder.encode(data)
            t = torch.randint(0, args.timesteps, (x_g.size(0),), device=device).long()

            loss = p_losses(
                denoise_model, x_g, t, data.stats, 
                sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, 
                loss_type="huber"
            )
            loss.backward()
            optimizer.step()

            train_loss_all += x_g.size(0) * loss.item()
            train_count    += x_g.size(0)

        mean_train_loss = train_loss_all / train_count

        denoise_model.eval()
        val_loss_all = 0.0
        val_count = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                x_g = autoencoder.encode(data)
                t = torch.randint(0, args.timesteps, (x_g.size(0),), device=device).long()

                loss = p_losses(
                    denoise_model, x_g, t, data.stats, 
                    sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, 
                    loss_type="huber"
                )
                val_loss_all += x_g.size(0) * loss.item()
                val_count    += x_g.size(0)

        mean_val_loss = val_loss_all / val_count

        train_loss_list_dn.append(mean_train_loss)
        val_loss_list_dn.append(mean_val_loss)

        if epoch % 5 == 0:
            dt_t = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print(f'{dt_t} Epoch: {epoch:04d}, Train Loss: {mean_train_loss:.5f}, Val Loss: {mean_val_loss:.5f}')

        scheduler.step()

        if best_val_loss > val_loss_all:
            best_val_loss = val_loss_all
            torch.save({
                'state_dict': denoise_model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, 'denoise_model.pth.tar')

    # Plot the denoiser losses
    epochs_range_dn = range(1, args.epochs_denoise+1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range_dn, train_loss_list_dn, label='Train Loss')
    plt.plot(epochs_range_dn, val_loss_list_dn, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Denoiser: Diffusion Model Loss')
    plt.legend()
    plt.savefig(f'Training/denoiser_loss_{time_suffix}.png')
    plt.close()

else:
    checkpoint = torch.load('denoise_model.pth.tar')
    denoise_model.load_state_dict(checkpoint['state_dict'])

denoise_model.eval()

# We can safely delete train_loader if you wish
del train_loader

# Compute MAE on validation set
mae = compute_mae(
    denoise_model, 
    autoencoder, 
    val_loader, 
    latent_dim=args.latent_dim, 
    timesteps=args.timesteps, 
    betas=betas, 
    device=device
)
print(f"Mean Absolute Error (MAE) on validation set: {mae:.4f}")

del val_loader

# Output for test set
with open("output.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["graph_id", "edge_list"])
    for k, data in enumerate(tqdm(test_loader, desc='Processing test set')):
        data = data.to(device)
        stat = data.stats
        bs = stat.size(0)
        graph_ids = data.filename

        samples = sample(
            denoise_model, data.stats, 
            latent_dim=args.latent_dim, 
            timesteps=args.timesteps, 
            betas=betas, 
            batch_size=bs
        )
        x_sample = samples[-1]
        adj = autoencoder.decode_mu(x_sample)
        stat_d = torch.reshape(stat, (-1, args.n_condition))

        for i in range(stat.size(0)):
            stat_x = stat_d[i].detach().cpu().numpy()
            Gs_generated = construct_nx_from_adj(adj[i,:,:].detach().cpu().numpy())
            graph_id = graph_ids[i]

            edge_list_text = ", ".join([f"({u}, {v})" for u, v in Gs_generated.edges()])
            writer.writerow([graph_id, edge_list_text])
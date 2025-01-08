import argparse
import os
import numpy as np
from datetime import datetime
import torch
from torch_geometric.loader import DataLoader
from sklearn.model_selection import ParameterGrid

from autoencoder import VariationalAutoEncoder
from denoise_model import DenoiseNN, p_losses
from utils import linear_beta_schedule, preprocess_dataset

# Fix seed for reproducibility
np.random.seed(13)

def train_and_evaluate(args, train_loader, val_loader):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Initialize VGAE model
    autoencoder = VariationalAutoEncoder(
        args.spectral_emb_dim + 1, args.hidden_dim_encoder, args.hidden_dim_decoder,
        args.latent_dim, args.n_layers_encoder, args.n_layers_decoder, args.n_max_nodes
    ).to(device)

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

    # Training loop for the autoencoder
    best_val_loss = np.inf
    for epoch in range(1, args.epochs_autoencoder + 1):
        autoencoder.train()
        train_loss_all = 0
        cnt_train = 0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            loss, _, _ = autoencoder.loss_function(data)
            loss.backward()
            train_loss_all += loss.item()
            cnt_train += 1
            optimizer.step()

        autoencoder.eval()
        val_loss_all = 0
        cnt_val = 0

        for data in val_loader:
            data = data.to(device)
            loss, _, _ = autoencoder.loss_function(data)
            val_loss_all += loss.item()
            cnt_val += 1

        avg_val_loss = val_loss_all / cnt_val

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

        scheduler.step()

    return best_val_loss

# Hyperparameter tuning function
def hyperparameter_tuning(trainset, valset, param_grid):
    results = []
    
    train_loader = DataLoader(trainset, batch_size=256, shuffle=True)
    val_loader = DataLoader(valset, batch_size=256, shuffle=False)

    for params in ParameterGrid(param_grid):
        print(f"Testing parameters: {params}")
        
        # Convert params to args-like object
        args = argparse.Namespace(**params)
        
        # Evaluate the configuration
        val_loss = train_and_evaluate(args, train_loader, val_loader)
        results.append({"params": params, "val_loss": val_loss})

    # Sort by validation loss
    results = sorted(results, key=lambda x: x["val_loss"])
    return results

if __name__ == "__main__":
    # Dataset preprocessing
    trainset = preprocess_dataset("train", n_max_nodes=50, spectral_emb_dim=10)
    valset = preprocess_dataset("valid", n_max_nodes=50, spectral_emb_dim=10)

    # Define the hyperparameter grid
    param_grid = {
        "lr": [1e-3, 1e-2],
        "hidden_dim_encoder": [64, 128],
        "hidden_dim_decoder": [256, 512],
        "latent_dim": [16, 32],
        "n_layers_encoder": [2, 3],
        "n_layers_decoder": [3, 4],
        "epochs_autoencoder": [100],  # Use fewer epochs for quick evaluation
        "epochs_denoise":[100], # Same because quick
        "n_max_nodes": [50],
        "spectral_emb_dim": [10]
    }

    # Perform hyperparameter tuning
    tuning_results = hyperparameter_tuning(trainset, valset, param_grid)

    # Save the best configuration
    best_config = tuning_results[0]
    print("Best hyperparameters:", best_config["params"])
    print("Best validation loss:", best_config["val_loss"])
    print("All results:", tuning_results)
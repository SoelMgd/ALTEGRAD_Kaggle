import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix

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

from autoencoder import VariationalAutoEncoder
from denoise_model import DenoiseNN, p_losses, sample
from utils import linear_beta_schedule, construct_nx_from_adj, preprocess_dataset, compute_graph_properties

from prop_predictor import PropertyPredictorGNN, GraphFeatures


from torch.utils.data import Subset
np.random.seed(13)

"""
Parses command line arguments for configuring the NeuralGraphGenerator model. This includes
settings for learning rates, architecture dimensions, training epochs, dropout rates, and 
parameters specific to the autoencoder (VGAE) and diffusion-based denoising model components.

Returns:
    argparse.Namespace: Parsed arguments as attributes for easy configuration of the model.
"""

# Argument parser
parser = argparse.ArgumentParser(description='NeuralGraphGenerator')

# Argument parser for configuring the NeuralGraphGenerator model
parser = argparse.ArgumentParser(description='Configuration for the NeuralGraphGenerator model')

# Learning rate for the optimizer
parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate for the optimizer, typically a small float value (default: 0.001)")

# Dropout rate
parser.add_argument('--dropout', type=float, default=0.0, help="Dropout rate (fraction of nodes to drop) to prevent overfitting (default: 0.0)")

# Batch size for training
parser.add_argument('--batch-size', type=int, default=256, help="Batch size for training, controlling the number of samples per gradient update (default: 256)")

# Number of epochs for the autoencoder training
parser.add_argument('--epochs-autoencoder', type=int, default=200, help="Number of training epochs for the autoencoder (default: 200)")

# Hidden dimension size for the encoder network
parser.add_argument('--hidden-dim-encoder', type=int, default=64, help="Hidden dimension size for encoder layers (default: 64)")

# Hidden dimension size for the decoder network
parser.add_argument('--hidden-dim-decoder', type=int, default=256, help="Hidden dimension size for decoder layers (default: 256)")

# Dimensionality of the latent space
parser.add_argument('--latent-dim', type=int, default=32, help="Dimensionality of the latent space in the autoencoder (default: 32)")

# Maximum number of nodes of graphs
parser.add_argument('--n-max-nodes', type=int, default=50, help="Possible maximum number of nodes in graphs (default: 50)")

# Number of layers in the encoder network
parser.add_argument('--n-layers-encoder', type=int, default=2, help="Number of layers in the encoder network (default: 2)")

# Number of layers in the decoder network
parser.add_argument('--n-layers-decoder', type=int, default=3, help="Number of layers in the decoder network (default: 3)")

# Dimensionality of spectral embeddings for graph structure representation
parser.add_argument('--spectral-emb-dim', type=int, default=10, help="Dimensionality of spectral embeddings for representing graph structures (default: 10)")

# Number of training epochs for the denoising model
parser.add_argument('--epochs-denoise', type=int, default=100, help="Number of training epochs for the denoising model (default: 100)")

# Number of timesteps in the diffusion
parser.add_argument('--timesteps', type=int, default=500, help="Number of timesteps for the diffusion (default: 500)")

# Hidden dimension size for the denoising model
parser.add_argument('--hidden-dim-denoise', type=int, default=512, help="Hidden dimension size for denoising model layers (default: 512)")

# Number of layers in the denoising model
parser.add_argument('--n-layers_denoise', type=int, default=3, help="Number of layers in the denoising model (default: 3)")

# Flag to toggle training of the autoencoder (VGAE)
parser.add_argument('--train-autoencoder', action='store_false', default=True, help="Flag to enable/disable autoencoder (VGAE) training (default: enabled)")

# Flag to toggle training of the diffusion-based denoising model
parser.add_argument('--train-denoiser', action='store_true', default=True, help="Flag to enable/disable denoiser training (default: enabled)")

# Dimensionality of conditioning vectors for conditional generation
parser.add_argument('--dim-condition', type=int, default=128, help="Dimensionality of conditioning vectors for conditional generation (default: 128)")

# Number of conditions used in conditional vector (number of properties)
parser.add_argument('--n-condition', type=int, default=7, help="Number of distinct condition properties used in conditional vector (default: 7)")

# Preprocess train data, validation data and test data. if put on True
parser.add_argument('--preprocess', action='store_true', default=False, help="Preprocess train data, validation data and test data. if put on True (default: False)")

# Use the old loss
parser.add_argument('--old_loss', action='store_true', default=False, help="Use the old loss (default: False)")

# alpha autoencoder (weight of the property loss)
parser.add_argument('--alpha_autoencoder', type=float, default=1.0, help="Alpha autoencoder (default: 1.0)")

# beta autoencoder (weight of the KL divergence)
parser.add_argument('--beta_autoencoder', type=float, default=0.05, help="Beta autoencoder (default: 0.05)")


args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# preprocess train data, validation data and test data. Only once for the first time that you run the code. Then the appropriate .pt files will be saved and loaded.
trainset = preprocess_dataset("train", args.n_max_nodes, args.spectral_emb_dim, args.preprocess)
validset = preprocess_dataset("valid", args.n_max_nodes, args.spectral_emb_dim, args.preprocess)
testset = preprocess_dataset("test", args.n_max_nodes, args.spectral_emb_dim, args.preprocess)



# initialize data loaders
train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(validset, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

# Count min and max for each of the 7 features
min_max = torch.zeros((7, 2))
for data in train_loader:
    for i in range(7):
        min_max[i, 0] = min(min_max[i, 0], torch.min(data.stats[:, i]))
        min_max[i, 1] = max(min_max[i, 1], torch.max(data.stats[:, i]))
min_max = min_max.to(device)

means = torch.mean(data.stats, dim=0).to(device)
stds = torch.std(data.stats, dim=0).to(device)

def generate_graph_from_features(features):
    """
    Génère une matrice d'adjacence à partir d'un vecteur de caractéristiques.

    Paramètres :
    - features : liste ou vecteur numpy contenant les 7 caractéristiques :
        [nombre de nœuds, nombre d'arêtes, degré moyen, nombre de triangles,
         coefficient de clustering global, maximum k-core, nombre de communautés]

    Retourne :
    - matrice d'adjacence (numpy array)
    """
    # Extraction des propriétés
    num_nodes, num_edges, avg_degree, num_triangles, clustering_coeff, max_kcore, num_communities = features
    num_nodes = int(num_nodes)
    num_edges = int(num_edges)
    num_triangles = int(num_triangles)
    max_kcore = int(max_kcore)
    num_communities = int(num_communities)  
    

    # Étape 1 : Initialisation du graphe
    G = nx.Graph()
    G.add_nodes_from(range(int(num_nodes)))

    # Étape 2 : Génération des arêtes
    while len(G.edges) < num_edges:
        u, v = np.random.choice(range(int(num_nodes)), size=2, replace=False)
        G.add_edge(u, v)

    # Étape 3 : Ajustement du clustering
    if clustering_coeff > 0:
        G = nx.watts_strogatz_graph(n=int(num_nodes), k=max(1, int(avg_degree)), p=clustering_coeff)

    # Étape 4 : Ajustement des triangles
    current_triangles = sum(nx.triangles(G).values()) // 3
    if current_triangles < num_triangles:
        for _ in range(int(num_triangles - current_triangles)):
            nodes = np.random.choice(range(int(num_nodes)), size=3, replace=False)
            G.add_edges_from([(nodes[0], nodes[1]), (nodes[1], nodes[2]), (nodes[0], nodes[2])])

    # Étape 5 : Ajustement du k-core
    current_kcore = nx.core_number(G)
    max_kcore_in_graph = max(current_kcore.values())
    if max_kcore_in_graph < max_kcore:
        for _ in range(max_kcore - max_kcore_in_graph):
            u, v = np.random.choice(range(int(num_nodes)), size=2, replace=False)
            G.add_edge(u, v)

    # Étape 6 : Ajustement des communautés
    if num_communities > 1:
        community_sizes = [int(num_nodes // num_communities) for _ in range(num_communities)]
        community_sizes[-1] += num_nodes % num_communities  # Ajuster la dernière communauté
        G = nx.stochastic_block_model(community_sizes, [[0.9 if i == j else 0.1 for j in range(num_communities)] for i in range(num_communities)])

    # Étape 7 : Retourner la matrice d'adjacence
    adj_matrix = nx.to_numpy_array(G)
    return adj_matrix

# Exemple d'utilisation
### Validation prop loss
val_prop_loss_sum = 0.0
val_prop_loss_true_sum = 0.0
val_count = 0

with torch.no_grad():
    for data in val_loader:
        data = data.to(device)
        stat = data.stats                 # (batch_size, 7)
        bs = stat.size(0)
        
        
        prop_pred_true = torch.zeros(bs, 7)

        for i in range(bs):
            adj = generate_graph_from_features(stat[i].cpu().numpy())
            prop_pred_true[i] = compute_graph_properties(adj[i])
            print(f'Graph {i} properties pred: {prop_pred_true[i]}')
            print(f'properties true: {stat[i]}')
            #print(adj[i,:,:])
            print('\n')
            

        # 4) On compare aux propriétés cibles (stat) en tenant compte de la normalisation
        prop_target_scaled = (stat - means) / stds
        prop_pred_true_scaled = (prop_pred_true - means) / stds

        # 5) On calcule une loss L1 (MAE) entre prop_est_scaled et prop_target_scaled
        prop_loss_batch_true = F.l1_loss(prop_pred_true_scaled, prop_target_scaled, reduction='mean')

        # On accumule pour faire la moyenne sur l'ensemble du set de validation
        val_prop_loss_true_sum += prop_loss_batch_true.item() * bs
        val_count += bs

val_prop_loss_mean = val_prop_loss_sum / val_count
val_prop_loss_mean_true = val_prop_loss_true_sum / val_count
print(f"[Validation] Property Loss (true) = {val_prop_loss_mean_true:.4f}")
###


# features = [14, 45, 6.4, 63, 0.56, 5, 2]  # Les 7 caractéristiques
# adj_matrix = generate_graph_from_features(features)
# print("Matrice d'adjacence générée :")
# print(adj_matrix)

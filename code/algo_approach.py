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


def generate_graph_from_features(features, max_iter=1000):
    """
    Génère une matrice d'adjacence à partir d'un vecteur de caractéristiques.
    
    Paramètres :
    -----------
    features : liste ou vecteur numpy contenant les 7 caractéristiques :
        [num_nodes, num_edges, avg_degree, num_triangles,
         clustering_coeff, max_kcore, num_communities]
         
    max_iter : nombre maximum d'itérations autorisées pour les ajustements.
    
    Retourne :
    ---------
    adj_matrix : numpy array
        La matrice d'adjacence du graphe généré qui tente de satisfaire 
        au mieux les contraintes.
    """
    # Déballage
    num_nodes, num_edges, avg_degree, num_triangles, clustering_coeff, max_kcore, num_communities = features
    
    num_nodes = int(num_nodes)
    num_edges = int(num_edges)
    avg_degree = float(avg_degree)
    num_triangles = int(num_triangles)
    max_kcore = int(max_kcore)
    num_communities = int(num_communities)

    # --- Sécurité de base ---
    # Nombre minimum d'arêtes: on ne peut pas être en dessous de 0
    num_edges = max(num_edges, 0)
    # Nombre maximum d'arêtes possible dans un graphe non orienté sans boucle
    max_possible_edges = num_nodes * (num_nodes - 1) // 2
    if num_edges > max_possible_edges:
        num_edges = max_possible_edges

    # 1) Génération initiale via SBM (Stochastic Block Model)
    #    On crée des blocs de taille similaire, et on fixe une proba plus élevée 
    #    à l'intérieur des blocs qu'entre blocs pour générer un effet communautaire.
    if num_communities < 1:
        num_communities = 1
    community_sizes = [num_nodes // num_communities] * num_communities
    # Ajuster pour qu'au total on ait bien num_nodes
    community_sizes[-1] += (num_nodes - sum(community_sizes))
    
    # Probas internes / externes : 
    # - plus la diagonal est élevée, plus les blocs forment des communautés denses
    # - p_out plus faible
    # On essaie d'adapter selon le clustering_coeff (idée : si clustering élevé, 
    # alors on met un p_inter plus bas et un p_intra plus haut).
    p_intra = min(0.8 + 0.2 * clustering_coeff, 1.0)  # Valeur haute si clustering élevé
    p_inter = max(0.05, 0.2 * (1 - clustering_coeff)) # Valeur basse si clustering élevé
    
    block_probs = []
    for i in range(num_communities):
        row = []
        for j in range(num_communities):
            row.append(p_intra if i == j else p_inter)
        block_probs.append(row)
    
    G = nx.stochastic_block_model(community_sizes, block_probs)
    
    # 2) Ajuster rapidement le nombre d'arêtes
    #    - Si trop d’arêtes : on en supprime aléatoirement
    #    - Si pas assez d’arêtes : on en ajoute aléatoirement
    def add_random_edge(G):
        # Ajoute une arête entre deux noeuds qui ne sont pas déjà connectés
        # (tentatives multiples si nécessaire)
        for _ in range(10):
            u, v = np.random.choice(G.nodes(), 2, replace=False)
            if not G.has_edge(u, v):
                G.add_edge(u, v)
                return
        
    def remove_random_edge(G):
        # Supprime une arête aléatoire
        edge = random.choice(list(G.edges()))
        G.remove_edge(*edge)
    
    current_edges = G.number_of_edges()
    while current_edges < num_edges:
        add_random_edge(G)
        current_edges = G.number_of_edges()
    while current_edges > num_edges:
        remove_random_edge(G)
        current_edges = G.number_of_edges()

    # 3) Ajuster le degré moyen, nombre de triangles et le clustering
    #    On fait du *rewiring* pour essayer d'augmenter/diminuer 
    #    le clustering ou le nombre de triangles sans trop casser 
    #    le nombre d’arêtes.
    
    def get_graph_stats(G):
        tri = sum(nx.triangles(G).values()) // 3
        c_global = nx.average_clustering(G)
        return tri, c_global
    
    # Petite fonction de "rewire"
    # On enlève une arête et on en ajoute une autre pour essayer de 
    # rapprocher le clustering / triangles de la cible.
    def edge_rewire(G):
        # On enlève une arête aléatoirement, puis on en ajoute une 
        # entre des nodes qui ne sont pas connectés
        edges = list(G.edges())
        if not edges:
            return
        edge_to_remove = random.choice(edges)
        G.remove_edge(*edge_to_remove)
        
        # Tenter d'ajouter
        add_random_edge(G)

    # On va faire un certain nombre d'itérations d'ajustement
    # en tâchant d'aller vers le num_triangles / clustering_coeff
    iteration = 0
    while iteration < max_iter:
        current_tri, current_clust = get_graph_stats(G)
        
        # Condition d'arrêt : on est "assez proche"
        # (difficile de faire exact, donc on met des tolérances)
        if (abs(current_tri - num_triangles) <= 2) and \
           (abs(current_clust - clustering_coeff) < 0.02):
            break
        
        # Si trop peu de triangles (vs. cible), 
        # on tente d'augmenter localement le clustering/triangles
        if current_tri < num_triangles:
            # On peut essayer d'ajouter manuellement des triangles
            # en identifiant 2 noeuds déjà connectés et un 3e 
            # relié à l'un mais pas à l'autre, etc.
            # Pour un code compact, on fait un rewire aléatoire 
            # qui *peut* générer plus de triangles
            edge_rewire(G)
        
        # Si trop de triangles
        elif current_tri > num_triangles:
            # Retirer un triangle (rewiring également, 
            # pour casser éventuellement un cycle)
            edge_rewire(G)
        
        # Ajustement du clustering
        if current_clust < clustering_coeff:
            # On tente un rewire en espérant augmenter le clustering
            edge_rewire(G)
        elif current_clust > clustering_coeff:
            # On tente un rewire, etc.
            edge_rewire(G)
        
        iteration += 1
    
    # 4) Ajustement du k-core : 
    #    Si le max_kcore dans le graphe est inférieur à la cible, 
    #    on essaie d’ajouter quelques arêtes pour augmenter 
    #    le degré de certains nœuds.
    #    (Ça peut être conflictuel avec le clustering ou le nb de triangles.)
    def ensure_kcore(G, target_k):
        c_numbers = nx.core_number(G)
        current_max_k = max(c_numbers.values())
        if current_max_k >= target_k:
            return
        else:
            # On essaie d'augmenter le k-core en ajoutant des arêtes 
            # autour des noeuds ayant déjà un degré élevé.
            # Cela n'est pas garanti de marcher, on fait un best-effort
            high_degree_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)
            top_nodes = [n for n, deg in high_degree_nodes[:min(5, len(high_degree_nodes))]]
            
            for _ in range(10):  # on tente 10 ajouts max
                u = random.choice(top_nodes)
                v = random.choice(top_nodes)
                if u != v and not G.has_edge(u, v):
                    G.add_edge(u, v)
                # recalculer vite fait
                c_numbers = nx.core_number(G)
                current_max_k = max(c_numbers.values())
                if current_max_k >= target_k:
                    break
    
    ensure_kcore(G, max_kcore)
    
    # 5) Dernier check du nombre d’arêtes (on peut s’être écarté) 
    #    à cause de l’étape k-core
    current_edges = G.number_of_edges()
    while current_edges < num_edges:
        add_random_edge(G)
        current_edges = G.number_of_edges()
    while current_edges > num_edges:
        remove_random_edge(G)
        current_edges = G.number_of_edges()
    
    # 6) Retourner la matrice d'adjacence
    adj_matrix = nx.to_numpy_array(G)
    return adj_matrix

# # Exemple d'utilisation
# ### Validation prop loss
# val_prop_loss_sum = 0.0
# val_prop_loss_true_sum = 0.0
# val_count = 0

# with torch.no_grad():
#     for data in val_loader:
#         data = data.to(device)
#         stat = data.stats                 # (batch_size, 7)
#         bs = stat.size(0)
        
        
#         prop_pred_true = torch.zeros(bs, 7)

#         for i in range(bs):
#             adj = generate_graph_from_features(stat[i].cpu().numpy())
#             prop_pred_true[i] = compute_graph_properties(torch.tensor(adj))
#             print(f'Graph {i} properties pred: {prop_pred_true[i]}')
#             print(f'properties true: {stat[i]}')
#             #print(adj[i,:,:])
#             print('\n')
            

#         # 4) On compare aux propriétés cibles (stat) en tenant compte de la normalisation
#         prop_target_scaled = (stat.cpu() - means.cpu()) / stds.cpu()
#         prop_pred_true_scaled = (prop_pred_true.cpu() - means.cpu()) / stds.cpu()

#         # 5) On calcule une loss L1 (MAE) entre prop_est_scaled et prop_target_scaled
#         prop_loss_batch_true = F.l1_loss(prop_pred_true_scaled, prop_target_scaled, reduction='mean')

#         # On accumule pour faire la moyenne sur l'ensemble du set de validation
#         val_prop_loss_true_sum += prop_loss_batch_true.item() * bs
#         val_count += bs

# val_prop_loss_mean = val_prop_loss_sum / val_count
# val_prop_loss_mean_true = val_prop_loss_true_sum / val_count
# print(f"[Validation] Property Loss (true) = {val_prop_loss_mean_true:.4f}")
# ###

# Save to a CSV file
with open("output_determinist.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    # Write the header
    writer.writerow(["graph_id", "edge_list"])
    
    prop_loss_sum = 0.0
    count = 0
    for k, data in enumerate(tqdm(test_loader, desc='Processing test set',)):

        stat = data.stats
        bs = stat.size(0)
        
        stat_gen = torch.zeros(bs, 7)

        graph_ids = data.filename
        
        for i in range(stat.size(0)):
            adj = generate_graph_from_features(stat[i].cpu().numpy())
            stat_gen[i] = compute_graph_properties(torch.tensor(adj))
            print(f'Graph {k*bs+i} properties: {stat_gen[i]}')
            print(f'properties true: {stat[i]}')
            

            Gs_generated = construct_nx_from_adj(adj)
            stat_x = stat_x.detach().cpu().numpy()

            # Define a graph ID
            graph_id = graph_ids[i] 
            
            print('\n')

            # Convert the edge list to a single string
            edge_list_text = ", ".join([f"({u}, {v})" for u, v in Gs_generated.edges()])           
            # Write the graph ID and the full edge list as a single row
            writer.writerow([graph_id, edge_list_text])
            
        stat_gen_scaled = (stat_gen - means.cpu()) / stds.cpu()
        stat_scaled = (stat - means.cpu()) / stds.cpu()
        prop_loss_batch = F.l1_loss(stat_gen_scaled, stat_scaled, reduction='mean')
        
        prop_loss_sum += prop_loss_batch.item() * bs
        count += bs
        
    prop_loss_mean = prop_loss_sum / count
    print(f"[Test] Property Loss = {prop_loss_mean:.4f}")
    
# features = [14, 45, 6.4, 63, 0.56, 5, 2]  # Les 7 caractéristiques
# adj_matrix = generate_graph_from_features(features)
# print("Matrice d'adjacence générée :")
# print(adj_matrix)

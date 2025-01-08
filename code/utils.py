import os
import math
import networkx as nx
import numpy as np
import scipy as sp
import scipy.sparse
import torch
import torch.nn.functional as F
import community as community_louvain

from torch import Tensor
from torch.utils.data import Dataset

from grakel.utils import graph_from_networkx
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from tqdm import tqdm
import scipy.sparse as sparse
from torch_geometric.data import Data

from extract_feats import extract_feats, extract_numbers



def preprocess_dataset(dataset, n_max_nodes, spectral_emb_dim, preprocess=False):

    data_lst = []
    if dataset == 'test':
        filename = './data/dataset_'+dataset+'.pt'
        desc_file = './data/'+dataset+'/test.txt'

        if os.path.isfile(filename) and not preprocess:
            data_lst = torch.load(filename)
            print(f'Dataset {filename} loaded from file')

        else:
            fr = open(desc_file, "r")
            for line in fr:
                line = line.strip()
                tokens = line.split(",")
                graph_id = tokens[0]
                desc = tokens[1:]
                desc = "".join(desc)
                feats_stats = extract_numbers(desc)
                feats_stats = torch.FloatTensor(feats_stats).unsqueeze(0)
                data_lst.append(Data(stats=feats_stats, filename = graph_id))
            fr.close()                    
            torch.save(data_lst, filename)
            print(f'Dataset {filename} saved')


    else:
        filename = './data/dataset_'+dataset+'.pt'
        graph_path = './data/'+dataset+'/graph'
        desc_path = './data/'+dataset+'/description'

        if os.path.isfile(filename) and not preprocess:
            data_lst = torch.load(filename)
            print(f'Dataset {filename} loaded from file')

        else:
            # traverse through all the graphs of the folder
            files = [f for f in os.listdir(graph_path)]
            adjs = []
            eigvals = []
            eigvecs = []
            n_nodes = []
            max_eigval = 0
            min_eigval = 0
            for fileread in tqdm(files):
                tokens = fileread.split("/")
                idx = tokens[-1].find(".")
                filen = tokens[-1][:idx]
                extension = tokens[-1][idx+1:]
                fread = os.path.join(graph_path,fileread)
                fstats = os.path.join(desc_path,filen+".txt")
                #load dataset to networkx
                if extension=="graphml":
                    G = nx.read_graphml(fread)
                    # Convert node labels back to tuples since GraphML stores them as strings
                    G = nx.convert_node_labels_to_integers(
                        G, ordering="sorted"
                    )
                else:
                    G = nx.read_edgelist(fread)
                # use canonical order (BFS) to create adjacency matrix
                ### BFS & DFS from largest-degree node

                
                CGs = [G.subgraph(c) for c in nx.connected_components(G)]

                # rank connected componets from large to small size
                CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)

                node_list_bfs = []
                for ii in range(len(CGs)):
                    node_degree_list = [(n, d) for n, d in CGs[ii].degree()]
                    degree_sequence = sorted(
                    node_degree_list, key=lambda tt: tt[1], reverse=True)

                    bfs_tree = nx.bfs_tree(CGs[ii], source=degree_sequence[0][0])
                    node_list_bfs += list(bfs_tree.nodes())

                adj_bfs = nx.to_numpy_array(G, nodelist=node_list_bfs)

                adj = torch.from_numpy(adj_bfs).float()
                diags = np.sum(adj_bfs, axis=0)
                diags = np.squeeze(np.asarray(diags))
                D = sparse.diags(diags).toarray()
                L = D - adj_bfs
                with sp.errstate(divide="ignore"):
                    diags_sqrt = 1.0 / np.sqrt(diags)
                diags_sqrt[np.isinf(diags_sqrt)] = 0
                DH = sparse.diags(diags).toarray()
                L = np.linalg.multi_dot((DH, L, DH))
                L = torch.from_numpy(L).float()
                eigval, eigvecs = torch.linalg.eigh(L)
                eigval = torch.real(eigval)
                eigvecs = torch.real(eigvecs)
                idx = torch.argsort(eigval)
                eigvecs = eigvecs[:,idx]

                edge_index = torch.nonzero(adj).t()

                size_diff = n_max_nodes - G.number_of_nodes()
                x = torch.zeros(G.number_of_nodes(), spectral_emb_dim+1)
                x[:,0] = torch.mm(adj, torch.ones(G.number_of_nodes(), 1))[:,0]/(n_max_nodes-1)
                mn = min(G.number_of_nodes(),spectral_emb_dim)
                mn+=1
                #print(f"eigvecs.shape: {eigvecs.shape}, spectral_emb_dim: {spectral_emb_dim}")

                x[:,1:mn] = eigvecs[:,:spectral_emb_dim]

                adj = F.pad(adj, [0, size_diff, 0, size_diff])
                adj = adj.unsqueeze(0)

                feats_stats = extract_feats(fstats)
                
                feats_stats = torch.FloatTensor(feats_stats).unsqueeze(0)
                #print(f"x.shape: {x.shape}")

                data_lst.append(Data(x=x, edge_index=edge_index, A=adj, stats=feats_stats, filename = filen))
            torch.save(data_lst, filename)
            print(f'Dataset {filename} saved')
    return data_lst


        

def construct_nx_from_adj(adj):
    G = nx.from_numpy_array(adj, create_using=nx.Graph)
    to_remove = []
    for node in G.nodes():
        if G.degree(node) == 0:
            to_remove.append(node)
    G.remove_nodes_from(to_remove)
    return G



def handle_nan(x):
    if math.isnan(x):
        return float(-100)
    return x




def masked_instance_norm2D(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-5):
    """
    x: [batch_size (N), num_objects (L), num_objects (L), features(C)]
    mask: [batch_size (N), num_objects (L), num_objects (L), 1]
    """
    mask = mask.view(x.size(0), x.size(1), x.size(2), 1).expand_as(x)
    mean = (torch.sum(x * mask, dim=[1,2]) / torch.sum(mask, dim=[1,2]))   # (N,C)
    var_term = ((x - mean.unsqueeze(1).unsqueeze(1).expand_as(x)) * mask)**2  # (N,L,L,C)
    var = (torch.sum(var_term, dim=[1,2]) / torch.sum(mask, dim=[1,2]))  # (N,C)
    mean = mean.unsqueeze(1).unsqueeze(1).expand_as(x)  # (N, L, L, C)
    var = var.unsqueeze(1).unsqueeze(1).expand_as(x)    # (N, L, L, C)
    instance_norm = (x - mean) / torch.sqrt(var + eps)   # (N, L, L, C)
    instance_norm = instance_norm * mask
    return instance_norm


def masked_layer_norm2D(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-5):
    """
    x: [batch_size (N), num_objects (L), num_objects (L), features(C)]
    mask: [batch_size (N), num_objects (L), num_objects (L), 1]
    """
    mask = mask.view(x.size(0), x.size(1), x.size(2), 1).expand_as(x)
    mean = torch.sum(x * mask, dim=[3,2,1]) / torch.sum(mask, dim=[3,2,1])   # (N)
    var_term = ((x - mean.view(-1,1,1,1).expand_as(x)) * mask)**2  # (N,L,L,C)
    var = (torch.sum(var_term, dim=[3,2,1]) / torch.sum(mask, dim=[3,2,1]))  # (N)
    mean = mean.view(-1,1,1,1).expand_as(x)  # (N, L, L, C)
    var = var.view(-1,1,1,1).expand_as(x)    # (N, L, L, C)
    layer_norm = (x - mean) / torch.sqrt(var + eps)   # (N, L, L, C)
    layer_norm = layer_norm * mask
    return layer_norm


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


import torch
import networkx as nx
import numpy as np

def compute_graph_properties(adj):
    """
    adj: torch.Tensor de forme (n_max_nodes, n_max_nodes)
    Retourne un tenseur de forme (7,) contenant : 
      [ nombre_de_noeuds, nombre_d_arêtes, degré_moyen, nb_triangles, clustering, max_kcore, nb_communautés ]
    """
    # On convertit la matrice en graph NetworkX
    # (en enlevant éventuellement les nœuds isolés pour éviter les indices hors bornes)
    G = nx.from_numpy_array(adj.detach().cpu().numpy())
    
    # Filtrage des nœuds isolés s’il y en a
    # (facultatif, selon la logique de votre dataset)
    isolated = list(nx.isolates(G))
    if len(isolated) > 0:
        G.remove_nodes_from(isolated)

    nb_nodes = G.number_of_nodes()
    nb_edges = G.number_of_edges()
    if nb_nodes > 1:
        avg_degree = float(np.mean([deg for (_, deg) in G.degree()]))
    else:
        avg_degree = 0.0
    
    # Triangles
    # NetworkX renvoie un dict {node: nb_triangles}, on somme pour avoir le total
    tri_dict = nx.triangles(G)
    nb_triangles = sum(tri_dict.values()) / 3  # chaque triangle est compté 3 fois
    
    # Clustering global
    clustering_coeff = nx.transitivity(G)  # ou nx.average_clustering(G)
    
    # k-core max
    # On récupère les k-cores jusqu’à exhaustion et on prend le plus grand k
    # (attention au cas où le graphe est petit)
    max_k = 0
    for k in range(1, nb_nodes+1):
        c = nx.k_core(G, k=k)
        if c.number_of_nodes() > 0:
            max_k = k
        else:
            break
    
    # Détection de communautés (ex: Louvain)
    # Nécessite la librairie python-louvain (community)
    try:
        import community as community_louvain
        partition = community_louvain.best_partition(G)
        nb_communities = len(set(partition.values()))
    except:
        # Par défaut, on considère qu’il n’y a qu’une seule composante
        nb_communities = 1
    
    prop_vec = [
        float(nb_nodes),
        float(nb_edges),
        float(avg_degree),
        float(nb_triangles),
        float(clustering_coeff),
        float(max_k),
        float(nb_communities),
    ]
    return torch.tensor(prop_vec, dtype=torch.float32, device=adj.device)

import random

def compute_graph_properties_approx(adj, edge_sampling_ratio=0.3):
    # Conversion en graph NetworkX
    G = nx.from_numpy_array(adj.detach().cpu().numpy())

    # Filtrage des nœuds isolés (optionnel)
    isolated = list(nx.isolates(G))
    if len(isolated) > 0:
        G.remove_nodes_from(isolated)

    nb_nodes = G.number_of_nodes()
    nb_edges = G.number_of_edges()
    if nb_nodes > 1:
        avg_degree = float(np.mean([deg for (_, deg) in G.degree()]))
    else:
        avg_degree = 0.0

    # Approx triangle count par sampling
    # On ne prend qu'une fraction edge_sampling_ratio des arêtes
    edges = list(G.edges())
    sampled_size = int(len(edges)*edge_sampling_ratio)
    if sampled_size < 1:
        sampled_size = 1
    sampled_edges = random.sample(edges, sampled_size)
    # On crée un mini-sous-graphe
    G_samp = nx.Graph()
    G_samp.add_nodes_from(G.nodes())
    G_samp.add_edges_from(sampled_edges)

    tri_dict = nx.triangles(G_samp)
    nb_triangles_samp = sum(tri_dict.values()) / 3
    # On extrapole
    nb_triangles = nb_triangles_samp / edge_sampling_ratio

    # Clustering global (c'est rapide, on peut le garder tel quel)
    clustering_coeff = nx.transitivity(G)

    # k-core max via core_number
    if nb_nodes > 0 and nb_edges>0:
        cnums = nx.core_number(G)  # renvoie un dict {node: coreVal}
        max_k = max(cnums.values()) if len(cnums) > 0 else 0
    else:
        max_k = 0

    # nb de communautés via Louvain (la partie chère)
    # => on peut la remplacer par un algo plus simple, ou skip
    try:
        import community as community_louvain
        partition = community_louvain.best_partition(G)
        nb_communities = len(set(partition.values()))
    except:
        nb_communities = 1
    
    prop_vec = [
        float(nb_nodes),
        float(nb_edges),
        float(avg_degree),
        float(nb_triangles),
        float(clustering_coeff),
        float(max_k),
        float(nb_communities),
    ]
    return torch.tensor(prop_vec, dtype=torch.float32, device=adj.device)

import torch
from torch_geometric.nn import GCNConv, global_add_pool
from torch import nn
import torch_geometric as pyg
import networkx as nx

class GraphFeatures(nn.Module):
    def __init__(self, eps=1e-6, community_threshold=0.01):
        """
        eps : petite constante pour éviter divisions par zéro.
        community_threshold : seuil pour compter une valeur propre 
                             comme "quasi-nulle" (pour approx. du nb de communautés).
        """
        super().__init__()
        self.eps = eps
        self.community_threshold = community_threshold
    
    def forward(self, A_batch: torch.Tensor):
        """
        Paramètres
        ----------
        A_batch : torch.Tensor de taille (B, n, n)
            B = batch_size, n = nombre maximal de noeuds
            Matrice(s) d'adjacence "soft" pour chaque échantillon du batch.
        
        Retour
        ------
        features : torch.Tensor de taille (B, 7)
            Contient, pour chaque échantillon du batch, les 7 propriétés suivantes :
              1) nombre_noeuds
              2) nombre_aretes_soft
              3) degre_moyen_soft
              4) nombre_triangles_soft
              5) clustering_global_soft
              6) max_kcore_soft (approx)
              7) nb_communautes_soft (approx)
        """
        B, n, _ = A_batch.shape  # (batch_size, n, n)

        # 1) Nombre de nœuds
        #    Ici, n est identique pour tout le batch (n max. de la reconstruction).
        #    Si on veut un tenseur par échantillon, on peut répéter n pour chaque batch.
        n_t = n * torch.ones(B, dtype=A_batch.dtype, device=A_batch.device)
        
        # 2) Nombre d'arêtes (soft)
        #    Pour chaque échantillon => somme(A[b,:,:]) / 2
        E_soft = 0.5 * A_batch.sum(dim=(1, 2))  # => [B]
        
        # 3) Degré moyen (soft)
        #    deg_mean_soft = (2 * E) / n
        deg_mean_soft = (2.0 * E_soft) / (n_t + self.eps)  # => [B]
        
        # 4) Nombre de triangles (soft) via A^3
        #    - On utilise bmm pour multiplication matricielle en batch
        A2 = torch.bmm(A_batch, A_batch)      # => [B, n, n]
        A3 = torch.bmm(A2, A_batch)          # => [B, n, n]
        #    - Trace en batch : diagonal(...).sum(dim=1)
        trace_A3 = torch.diagonal(A3, dim1=1, dim2=2).sum(dim=1)  # => [B]
        nb_triangles_soft = trace_A3 / 6.0                         # => [B]
        
        # 5) Coefficient de clustering global (soft)
        #    deg[b, i] = somme_j A_batch[b, i, j]
        deg = A_batch.sum(dim=2)  # => [B, n]
        #    nb de triplets connectés soft pour l'échantillon b : 0.5 * sum_i deg_i(deg_i - 1)
        triplets = 0.5 * (deg * (deg - 1.0)).sum(dim=1)  # => [B]
        #    clustering_global = 3 * nb_triangles / nb_triplets
        clustering_soft = (3.0 * nb_triangles_soft) / (triplets + self.eps)  # => [B]
        
        # 6) Max k-core (approx) = max degré (soft)
        #    (dans la vraie définition, c'est discret & itératif)
        max_kcore_soft = deg.max(dim=1).values  # => [B]
        
        # 7) Nombre de communautés (approx)
        #    - On calcule le Laplacien L_b = D_b - A_batch[b] pour chaque b
        #    - On prend ses valeurs propres, et on compte celles < community_threshold en abso
        nb_coms_list = []
        for b in range(B):
            D_b = torch.diag(deg[b])        # diag(deg_i) => [n, n]
            L_b = D_b - A_batch[b]         # => [n, n]
            vals_b = torch.linalg.eigvalsh(L_b)   # => [n]
            # on compte combien de vp sont proches de 0
            nb_coms_b = (vals_b.abs() < self.community_threshold).sum()
            nb_coms_list.append(nb_coms_b)
        
        nb_coms_soft = torch.stack(nb_coms_list, dim=0).to(A_batch.dtype)  # => [B]
        
        # On empile nos 7 propriétés en un seul tenseur [B, 7]
        features = torch.stack([
            n_t, 
            E_soft, 
            deg_mean_soft, 
            nb_triangles_soft, 
            clustering_soft,
            max_kcore_soft,
            nb_coms_soft
        ], dim=1)  # => [B, 7]

        return features

class PropertyPredictorGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        input_dim: Dimension des features des nœuds (par exemple 1 si features par défaut)
        hidden_dim: Taille des couches cachées
        output_dim: Nombre de propriétés à prédire
        """
        super(PropertyPredictorGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, adj_matrix):
        """
        adj_matrix: torch.Tensor (n_nodes, n_nodes)
            Matrice d'adjacence d'un graphe
        """
        # Conversion en format edge_index (requis par PyTorch Geometric)
        edge_index = self.adj_to_edge_index(adj_matrix)
        n_nodes = adj_matrix.size(0)

        # Initialiser les features des nœuds (par exemple, tous les nœuds avec feature 1)
        x = torch.ones((n_nodes, 1), device=adj_matrix.device)  # Feature par défaut = 1

        # GCN Forward Pass
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()

        # Pooling global pour obtenir un vecteur global pour le graphe
        batch = torch.zeros(n_nodes, dtype=torch.long, device=adj_matrix.device)  # Tous les nœuds sont dans le même graphe
        x = global_add_pool(x, batch)

        # Passer à travers le fully connected pour produire les propriétés
        out = self.fc(x)
        return out

    @staticmethod
    def adj_to_edge_index(adj_matrix):
        """
        Convertit une matrice d'adjacence en edge_index
        """
        edge_index = torch.nonzero(adj_matrix > 0, as_tuple=False).t().contiguous()
        return edge_index

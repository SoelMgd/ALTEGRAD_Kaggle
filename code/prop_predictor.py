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
    
    def forward(self, A: torch.Tensor):
        """
        Paramètres
        ----------
        A : torch.Tensor de taille (n, n)
            Matrice d'adjacence "soft" (symétrique, sans boucles : Aii=0).
        
        Retour
        ------
        features : torch.Tensor de taille (7,)
            [ nombre_noeuds,
              nombre_aretes_soft,
              degre_moyen_soft,
              nombre_triangles_soft,
              clustering_global_soft,
              max_kcore_soft (approximé),
              nb_communautes_soft (approximé) ]
        """
        
        # 1) Nombre de noeuds
        n = A.shape[0]
        n_t = torch.tensor([n], dtype=A.dtype, device=A.device)  # constant “tensorisé”
        
        # 2) Nombre d'arêtes (soft)
        #    Diviser par 2 car on suppose A symétrique => chaque arête compte deux fois (i->j, j->i)
        E_soft = 0.5 * A.sum()
        
        # 3) Degré moyen (soft)
        deg_mean_soft = E_soft * 2.0 / (n_t + self.eps)  # = (A.sum()/n)
        
        # 4) Nombre de triangles (soft) = trace(A^3)/6
        #    On calcule A^2, puis A^3
        A2 = A @ A
        A3 = A2 @ A
        nb_triangles_soft = torch.trace(A3) / 6.0
        
        # 5) Coefficient de clustering global (soft)
        #    deg_i = somme_j A[i,j]
        deg = A.sum(dim=1)
        #    nombre de triplets connectés soft = sum_i (deg_i * (deg_i - 1)) / 2
        triplets_soft = 0.5 * torch.sum(deg * (deg - 1.0))
        
        #    clustering global = (3 * nb_triangles_soft) / (nb_triplets + eps)
        #    (NB: "3 x triangles" = nombre de triplets fermés)
        clustering_soft = (3.0 * nb_triangles_soft) / (triplets_soft + self.eps)
        
        # 6) Max k-core (approx) = max degré (soft)
        #    (Vraie définition non différenciable. Ici, on approxime en prenant le max du degré)
        max_kcore_soft = torch.max(deg)
        
        # 7) Nombre de communautés (approx)
        #    On utilise la matrice Laplacien L = D - A où D=diag(deg)
        #    Nombre de composantes connexes = nb de vp = 0 (classique).
        #    On va approximer le "nb de communautés" 
        #    en comptant le nombre de vp < community_threshold en valeur absolue.
        D = torch.diag(deg)
        L = D - A
        #    Calcul des valeurs propres (pour petit n)
        #    Pour du big graph, on ferait autrement.
        vals = torch.linalg.eigvalsh(L)  # valeurs propres réelles (L symétrique)
        
        #    Comptage “soft” : on considère qu'une vp < community_threshold => “une communauté”
        #    Comme c'est un test direct, ça reste un peu "discret".
        #    On peut faire un "smooth counting" via une sigmoïde autour du threshold, par ex :
        #       count_soft = sum( sigmoid( (threshold - lambda_i)/beta ) )
        #    Pour la démo, on fait un “step” direct, un peu moins diff'.
        nb_coms_soft = (vals.abs() < self.community_threshold).sum()  # compte entier
        #    Pour le rendre float/diff, on le cast en float:
        nb_coms_soft = nb_coms_soft.to(A.dtype)
        
        # On empile tout
        features = torch.stack([
            n_t, 
            E_soft, 
            deg_mean_soft, 
            nb_triangles_soft, 
            clustering_soft,
            max_kcore_soft,
            nb_coms_soft
        ])
        
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

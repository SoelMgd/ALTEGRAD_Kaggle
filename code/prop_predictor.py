import torch
from torch_geometric.nn import GCNConv, global_add_pool
from torch import nn
import torch_geometric as pyg
import networkx as nx

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

from torch_geometric.nn import GCNConv, global_add_pool

class PropertyPredictorGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PropertyPredictorGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = global_add_pool(x, batch)  # Agrégation par graphe
        out = self.fc(x)
        return out

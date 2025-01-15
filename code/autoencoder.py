import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from torch_geometric.nn import global_add_pool

##########################
#    GRAPH AUGMENTATION
##########################
def augment_graph(data, drop_edge_prob=0.1, drop_feat_prob=0.1):
    """
    A simple example of augmenting a PyG Data object:
      - Randomly drop edges
      - Randomly mask node features

    data: PyG Data object
    drop_edge_prob: probability of removing each edge
    drop_feat_prob: probability of masking each node feature dimension
    Returns a new Data object with slightly modified adjacency and features.
    """
    # 1) Edge Drop
    edge_index = data.edge_index.clone()
    num_edges = edge_index.size(1)
    mask = torch.rand(num_edges, device=edge_index.device) > drop_edge_prob
    edge_index = edge_index[:, mask]  # keep only some edges

    # 2) Feature Mask
    x = data.x.clone()
    # x is typically shape (num_nodes, num_features)
    mask_feats = torch.rand_like(x) < drop_feat_prob
    x[mask_feats] = 0.0  # simple approach: zero out

    # create a new Data object with the augmented adjacency/feature
    aug_data = data.clone()
    aug_data.edge_index = edge_index
    aug_data.x = x
    return aug_data


#############################
#     DECODER
#############################
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_layers, n_nodes):
        super().__init__()
        self.n_layers = n_layers
        self.n_nodes = n_nodes

        mlp_layers = [nn.Linear(latent_dim, hidden_dim)]
        for i in range(n_layers-2):
            mlp_layers.append(nn.Linear(hidden_dim, hidden_dim))
        mlp_layers.append(nn.Linear(hidden_dim, 2*n_nodes*(n_nodes-1)//2))

        self.mlp = nn.ModuleList(mlp_layers)
        self.relu = nn.ReLU()

    def forward(self, z):
        for i in range(self.n_layers-1):
            z = self.relu(self.mlp[i](z))
        z = self.mlp[self.n_layers-1](z)
        # reshape and gumbel softmax as before
        z = torch.reshape(z, (z.size(0), -1, 2))
        z = F.gumbel_softmax(z, tau=1.0, hard=True)[:, :, 0]

        adj = torch.zeros(z.size(0), self.n_nodes, self.n_nodes, device=z.device)
        idx = torch.triu_indices(self.n_nodes, self.n_nodes, 1)
        adj[:, idx[0], idx[1]] = z
        adj = adj + torch.transpose(adj, 1, 2)
        return adj


#############################
#     GIN ENCODER
#############################
class GINEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()

        # first layer
        self.convs.append(
            GINConv(
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.LeakyReLU(0.2),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LeakyReLU(0.2)
                )
            )
        )
        # additional layers
        for _ in range(n_layers-1):
            self.convs.append(
                GINConv(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.LeakyReLU(0.2),
                        nn.BatchNorm1d(hidden_dim),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.LeakyReLU(0.2)
                    )
                )
            )

        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc_latent = nn.Linear(hidden_dim, latent_dim)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.dropout(x, p=self.dropout, training=self.training)

        out = global_add_pool(x, data.batch)
        out = self.bn(out)
        z = self.fc_latent(out)
        return z


#############################
#     CONTRASTIVE VAE
#############################
class VariationalAutoEncoder(nn.Module):
    def __init__(
        self, input_dim, hidden_dim_enc, hidden_dim_dec, latent_dim,
        n_layers_enc, n_layers_dec, n_max_nodes
    ):
        super().__init__()

        # Encoder (GIN)
        self.encoder = GINEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim_enc,
            latent_dim=latent_dim,
            n_layers=n_layers_enc
        )

        # VAE heads: mu/logvar
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)

        # Decoder
        self.decoder = Decoder(latent_dim, hidden_dim_dec, n_layers_dec, n_max_nodes)

        # Condition MLP (text -> latent_dim)
        self.condition_mlp = nn.Sequential(
            nn.Linear(7, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

        # -------------------------
        #   PROJECTION HEADS
        # -------------------------
        # 1) For graph embeddings z
        self.graph_proj_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        # 2) For text embeddings c
        self.text_proj_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

    # ------------------------------------
    #          ENCODE/DECODE
    # ------------------------------------
    def encode(self, data):
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        z = self.reparameterize(mu, logvar)
        return z

    def reparameterize(self, mu, logvar):
        if self.training:
            std = (0.5 * logvar).exp_()
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, mu, logvar):
        z = self.reparameterize(mu, logvar)
        adj = self.decoder(z)
        return adj

    def decode_mu(self, mu):
        return self.decoder(mu)

    # ------------------------------------
    #         CONTRASTIVE LOSSES
    # ------------------------------------
    def graph_graph_contrastive_loss(self, z1, z2, temperature=0.07):
        """
        Compare z1 and z2 as a positive pair, with in-batch negatives.
        We assume z1, z2 => shape (B, latent_dim).
        We'll do an InfoNCE style approach.
        """
        # L2-normalize for stable training => dot product ~ cosine similarity
        z1_norm = F.normalize(z1, dim=-1)  # (B, latent_dim)
        z2_norm = F.normalize(z2, dim=-1)  # (B, latent_dim)

        # Similarity matrix: (2B x 2B), but let's just do (B,B) for z1 vs z2
        sim_matrix = torch.matmul(z1_norm, z2_norm.t())  # (B,B)

        # Scale by temperature
        sim_matrix = sim_matrix / temperature

        # Targets are diagonal => same index
        batch_size = z1.size(0)
        labels = torch.arange(batch_size, device=z1.device)

        # Cross-entropy (InfoNCE)
        # pos => diagonal elements of sim_matrix
        # This is a simplistic approach: we treat (z1_i, z2_i) as positives,
        # and (z1_i, z2_j) for j != i as negatives.
        loss = F.cross_entropy(sim_matrix, labels)
        return loss

    def graph_text_contrastive_loss(self, z, c_emb, temperature=0.07):
        """
        Compare graph embedding z to text embedding c_emb as a positive pair.
        Similar InfoNCE approach, in-batch negatives.
        """
        z_norm = F.normalize(z, dim=-1)
        c_norm = F.normalize(c_emb, dim=-1)
        sim_matrix = torch.matmul(z_norm, c_norm.t())  # (B,B)
        sim_matrix = sim_matrix / temperature

        batch_size = z.size(0)
        labels = torch.arange(batch_size, device=z.device)
        loss = F.cross_entropy(sim_matrix, labels)
        return loss

    # ------------------------------------
    #        VAE LOSS + GCL
    # ------------------------------------
    def loss_function(
        self, data,
        alpha_g2g=0.1,         # weight for graph–graph contrast
        alpha_g2t=0.1,         # weight for graph–text contrast
        beta=0.05,             # weight for KLD
        temperature=0.07,
        drop_edge_prob=0.1,
        drop_feat_prob=0.1
    ):
        """
        Compute a combined loss:
          - VAE reconstruction + KLD using *one* standard (unaugmented) view
          - Graph–Graph Contrast: using two augmented views
          - Graph–Text Contrast: use the *original* or one augmented view for the text
        """

        # ----------------------------------------
        # 1) Graph–Graph multi-view augmentation
        # ----------------------------------------
        view1_data = augment_graph(data, drop_edge_prob, drop_feat_prob)
        view2_data = augment_graph(data, drop_edge_prob, drop_feat_prob)

        # Encode each augmented view
        z1_pre = self.encoder(view1_data)
        z2_pre = self.encoder(view2_data)

        # Reparam for each (i.e., VAE style), or for simplicity,
        # we can treat them as deterministic for contrast only.
        mu1 = self.fc_mu(z1_pre)
        logvar1 = self.fc_logvar(z1_pre)
        z1 = self.reparameterize(mu1, logvar1)

        mu2 = self.fc_mu(z2_pre)
        logvar2 = self.fc_logvar(z2_pre)
        z2 = self.reparameterize(mu2, logvar2)

        # Project them for graph–graph contrast
        z1_proj = self.graph_proj_head(z1)  # shape (B, latent_dim)
        z2_proj = self.graph_proj_head(z2)

        # Graph–Graph contrastive loss
        g2g_loss = self.graph_graph_contrastive_loss(z1_proj, z2_proj, temperature=temperature)

        # ----------------------------------------
        # 2) VAE reconstruction + KLD (one view)
        #    We can pick the *original* data or view1_data for reconstruction
        # ----------------------------------------
        x_g = self.encoder(data)  # or use view1_data if you want
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        z = self.reparameterize(mu, logvar)

        # Decode => adjacency
        adj_pred = self.decoder(z)
        recon_loss = F.l1_loss(adj_pred, data.A, reduction='mean')

        # KLD
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # ----------------------------------------
        # 3) Graph–Text contrastive
        #    We'll use the same z as above (the one used for recon),
        #    or we could choose z1 if you prefer.
        # ----------------------------------------
        # Condition MLP => c
        c = self.condition_mlp(data.stats.float())
        # Project text => c'
        c_proj = self.text_proj_head(c)
        # Project graph => z'
        z_proj = self.graph_proj_head(z)
        # Graph–Text contrastive loss
        g2t_loss = self.graph_text_contrastive_loss(z_proj, c_proj, temperature=temperature)

        # ----------------------------------------
        # Combine all
        # ----------------------------------------
        total_loss = recon_loss + beta * kld_loss
        total_loss += alpha_g2g * g2g_loss
        total_loss += alpha_g2t * g2t_loss

        return total_loss, recon_loss, kld_loss, g2g_loss, g2t_loss
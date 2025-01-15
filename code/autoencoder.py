import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool

##########################
#    SIMPLE AUTOENCODER
#    with Graph→Text Contrast
##########################

class GINEncoder(nn.Module):
    """
    GIN-based graph encoder:
      - Input dimension: 'input_dim'
      - Hidden dimension: 'hidden_dim'
      - Latent dimension: 'latent_dim'
      - 'n_layers' GINConv layers
      - 'dropout' for regularization
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()

        # First GIN layer
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

        # Additional GIN layers
        for _ in range(n_layers - 1):
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

        # GIN layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pool -> BN -> project to latent
        out = global_add_pool(x, data.batch)
        out = self.bn(out)
        z = self.fc_latent(out)
        return z


class Decoder(nn.Module):
    """
    Simple MLP-based decoder that produces a discrete adjacency matrix
    via Gumbel-Softmax.
    """
    def __init__(self, latent_dim, hidden_dim, n_layers, n_nodes):
        super().__init__()
        self.n_layers = n_layers
        self.n_nodes = n_nodes

        mlp_layers = [nn.Linear(latent_dim, hidden_dim)]
        for _ in range(n_layers - 2):
            mlp_layers.append(nn.Linear(hidden_dim, hidden_dim))
        mlp_layers.append(nn.Linear(hidden_dim, 2 * n_nodes * (n_nodes - 1) // 2))

        self.mlp = nn.ModuleList(mlp_layers)
        self.relu = nn.ReLU()

    def forward(self, z):
        # MLP layers
        for i in range(self.n_layers - 1):
            z = self.relu(self.mlp[i](z))
        z = self.mlp[self.n_layers - 1](z)

        # Reshape and apply Gumbel-Softmax
        z = torch.reshape(z, (z.size(0), -1, 2))
        z = F.gumbel_softmax(z, tau=1.0, hard=True)[:, :, 0]

        # Build adjacency (upper triangular + transpose)
        adj = torch.zeros(z.size(0), self.n_nodes, self.n_nodes, device=z.device)
        idx = torch.triu_indices(self.n_nodes, self.n_nodes, 1)
        adj[:, idx[0], idx[1]] = z
        adj = adj + torch.transpose(adj, 1, 2)
        return adj


class VariationalAutoEncoder(nn.Module):
    """
    Simple VGAE with:
      - GINEncoder for graphs
      - MLP Decoder for adjacency
      - Graph→Text contrastive loss in the latent space
    """
    def __init__(
        self,
        input_dim,
        hidden_dim_enc,
        hidden_dim_dec,
        latent_dim,
        n_layers_enc,
        n_layers_dec,
        n_max_nodes
    ):
        super().__init__()
        # Encoder
        self.encoder = GINEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim_enc,
            latent_dim=latent_dim,
            n_layers=n_layers_enc
        )
        # VAE heads (mu / logvar)
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)

        # Decoder
        self.decoder = Decoder(
            latent_dim, hidden_dim_dec, n_layers_dec, n_max_nodes
        )

        # Condition MLP (text → latent_dim)
        self.condition_mlp = nn.Sequential(
            nn.Linear(7, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

        # Projection heads (for contrastive alignment)
        self.graph_proj_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        self.text_proj_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

    # -----------
    # VAE Core
    # -----------
    def encode(self, data):
        """
        Encode a PyG Data object into a latent z via GINEncoder -> mu/logvar -> reparam.
        """
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        z = self.reparameterize(mu, logvar)
        return z

    def reparameterize(self, mu, logvar):
        """
        Standard VAE reparam trick:
          z = mu + sigma * eps
        """
        if self.training:
            std = (0.5 * logvar).exp_()
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, mu, logvar):
        """
        Decode from mu/logvar -> reparam -> adjacency
        """
        z = self.reparameterize(mu, logvar)
        return self.decoder(z)

    def decode_mu(self, mu):
        """
        Directly decode from a latent vector mu.
        """
        return self.decoder(mu)

    # -----------
    # Contrastive Loss (Graph→Text)
    # -----------
    def contrastive_loss(self, z, c_emb, temperature=0.07):
        """
        Graph→Text contrast: InfoNCE with in-batch negatives.
          z, c_emb: shape (B, latent_dim)
        """
        z_norm = F.normalize(z, dim=-1)
        c_norm = F.normalize(c_emb, dim=-1)

        # Similarity matrix (B, B)
        sim_matrix = torch.matmul(z_norm, c_norm.t())
        sim_matrix /= temperature

        batch_size = z.size(0)
        labels = torch.arange(batch_size, device=z.device)

        # Cross-entropy (diagonal are positives)
        loss = F.cross_entropy(sim_matrix, labels)
        return loss

    # -----------
    # Full Loss Function
    # -----------
    def loss_function(
        self,
        data,
        beta=0.05,      # Weight for KLD
        alpha=0.1,      # Weight for contrastive loss
        temperature=0.07
    ):
        """
        Overall loss = Recon + Beta*KLD + Alpha*Contrastive
          - Recon via L1 adjacency difference
          - Contrastive via z vs. c_emb
        """
        # 1) Encode => mu, logvar => reparam
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        z = self.reparameterize(mu, logvar)

        # 2) Decode => adjacency => reconstruction loss
        adj_pred = self.decoder(z)
        recon_loss = F.l1_loss(adj_pred, data.A, reduction='mean')

        # 3) KLD
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # 4) Contrastive (Graph → Text)
        # Map text stats to embedding
        c = self.condition_mlp(data.stats.float())   # shape (B, latent_dim)
        # Project both
        z_proj = self.graph_proj_head(z)
        c_proj = self.text_proj_head(c)
        cont_loss = self.contrastive_loss(z_proj, c_proj, temperature=temperature)

        # Combine
        total_loss = recon_loss + beta * kld_loss + alpha * cont_loss

        return total_loss, recon_loss, kld_loss, cont_loss
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GINConv
from torch_geometric.nn import global_add_pool

import numpy as np

from utils import compute_graph_properties, compute_graph_properties_approx



# Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_layers, n_nodes):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.n_nodes = n_nodes

        mlp_layers = [nn.Linear(latent_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) for i in range(n_layers-2)]
        mlp_layers.append(nn.Linear(hidden_dim, 2*n_nodes*(n_nodes-1)//2))

        self.mlp = nn.ModuleList(mlp_layers)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for i in range(self.n_layers-1):
            x = self.relu(self.mlp[i](x))
        
        x = self.mlp[self.n_layers-1](x)
        x = torch.reshape(x, (x.size(0), -1, 2))
        x = F.gumbel_softmax(x, tau=1, hard=True)[:,:,0]

        adj = torch.zeros(x.size(0), self.n_nodes, self.n_nodes, device=x.device)
        idx = torch.triu_indices(self.n_nodes, self.n_nodes, 1)
        adj[:,idx[0],idx[1]] = x
        adj = adj + torch.transpose(adj, 1, 2)
        return adj




class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim),  
                            nn.LeakyReLU(0.2),
                            nn.BatchNorm1d(hidden_dim),
                            nn.Linear(hidden_dim, hidden_dim), 
                            nn.LeakyReLU(0.2))
                            ))                        
        for layer in range(n_layers-1):
            self.convs.append(GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim),  
                            nn.LeakyReLU(0.2),
                            nn.BatchNorm1d(hidden_dim),
                            nn.Linear(hidden_dim, hidden_dim), 
                            nn.LeakyReLU(0.2))
                            )) 

        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, latent_dim)
        

    def forward(self, data):
        edge_index = data.edge_index
        x = data.x
        #print(f"x.shape: {x.shape}, edge_index.shape: {edge_index.shape}")

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.dropout(x, self.dropout, training=self.training)

        out = global_add_pool(x, data.batch)
        out = self.bn(out)
        out = self.fc(out)
        return out


# Variational Autoencoder
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim_enc, hidden_dim_dec, latent_dim, n_layers_enc, n_layers_dec, n_max_nodes, predictor):
        super(VariationalAutoEncoder, self).__init__()
        self.n_max_nodes = n_max_nodes
        self.input_dim = input_dim
        self.encoder = GIN(input_dim, hidden_dim_enc, hidden_dim_enc, n_layers_enc)
        self.fc_mu = nn.Linear(hidden_dim_enc, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim_enc, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim_dec, n_layers_dec, n_max_nodes)
        self.predicator = predictor

    def forward(self, data):
        #print(f"data.x.shape: {data.x.shape}, input_dim: {self.encoder.convs[0].nn[0].in_features}")
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        adj = self.decoder(x_g)
        return adj

    def encode(self, data):
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        return x_g

    def reparameterize(self, mu, logvar, eps_scale=1.):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, mu, logvar):
       x_g = self.reparameterize(mu, logvar)
       adj = self.decoder(x_g)
       return adj

    def decode_mu(self, mu):
       adj = self.decoder(mu)
       return adj

    def loss_function_old(self, data, beta=0.05):
        x_g  = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        adj = self.decoder(x_g)
        
        recon = F.l1_loss(adj, data.A, reduction='mean')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon + beta*kld

        return loss, recon, kld

    def loss_function(self, data, means, stds, beta=0.05, alpha=1.0):
        """
        data: batch issu du DataLoader
        min_max: Tensor contenant les min et max de chaque propriété pour normalisation
        beta: pondère la partie KLD
        alpha: pondère la partie property loss
        """
        # Encodage
        x_g  = self.encoder(data)                # [batch_size, hidden_dim_enc]
        mu = self.fc_mu(x_g)                     # [batch_size, latent_dim]
        logvar = self.fc_logvar(x_g)             # [batch_size, latent_dim]
        z = self.reparameterize(mu, logvar)      # [batch_size, latent_dim]

        # Reconstruction d’adjacence
        adj_recon = self.decoder(z)              # [batch_size, n_max_nodes, n_max_nodes]

        # 1) Perte de reconstruction (ici L1, vous pouvez essayer L2, BCE, etc.)
        recon_loss = F.l1_loss(adj_recon, data.A, reduction='mean')

        # 2) KLD du VAE
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kld = kld / adj_recon.size(0)  # Normaliser par batch size

        # 3) Property loss sur tout le batch
        # Normalisation pour éviter de dépasser les bornes min/max

        # Prédiction des propriétés pour tout le batch
        prop_est = self.predicator(adj_recon)  # [batch_size, 7]
        prop_est_scaled = (prop_est - means) / stds
        prop_target = data.stats
        prop_target_scaled = (prop_target - means) / stds

        # Calcul de la loss (MAE entre propriétés prédictes et cibles)
        prop_loss = F.l1_loss(prop_est_scaled, prop_target_scaled, reduction='mean')

        # Combine les pertes
        loss = recon_loss + beta * kld + alpha * prop_loss

        return loss, recon_loss, kld, prop_loss

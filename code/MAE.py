import torch
import torch.nn.functional as F
import networkx as nx
from torch_geometric.data import DataLoader
from denoise_model import DenoiseNN, p_losses, sample
from tqdm import tqdm
import numpy as np

# Function to calculate graph statistics
def calculate_graph_statistics(graph):
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    average_degree = 2 * n_edges / n_nodes if n_nodes > 0 else 0
    n_triangles = sum(nx.triangles(graph).values()) // 3
    global_clustering_coefficient = nx.transitivity(graph)
    graph_maximum_k_core = max(nx.core_number(graph).values())
    n_communities = len(list(nx.connected_components(graph)))

    return [
        n_nodes, 
        n_edges, 
        average_degree, 
        n_triangles, 
        global_clustering_coefficient, 
        graph_maximum_k_core, 
        n_communities
    ]

# Function to compute MAE
def compute_mae(denoiser, autoencoder, val_loader, latent_dim, timesteps, betas, device):
    denoiser.eval()
    mae_total = 0
    num_graphs = 0

    with torch.no_grad():
        for data in tqdm(val_loader, desc="Evaluating MAE"):
            data = data.to(device)

            # Perform inference
            generated_samples = sample(
                denoiser, data.stats, latent_dim=latent_dim, timesteps=timesteps, betas=betas,
                batch_size=data.stats.size(0)
            )
            x_sample = generated_samples[-1]

            # Decode adjacency matrices
            adj_matrices = autoencoder.decode_mu(x_sample)

            for i in range(data.stats.size(0)):
                # Construct graph from adjacency matrix
                adj_matrix = adj_matrices[i].detach().cpu().numpy()
                graph = nx.from_numpy_array((adj_matrix > 0.5).astype(int))

                # Calculate statistics of generated graph
                generated_stats = calculate_graph_statistics(graph)

                # Compare with ground truth statistics
                true_stats = data.stats[i].detach().cpu().numpy()
                mae_total += np.abs(np.array(generated_stats) - true_stats).mean()
                num_graphs += 1

    return mae_total / num_graphs

if __name__ == "__main__":
    # Example usage
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load preprocessed validation set
    validset = preprocess_dataset("valid", n_max_nodes=50, spectral_emb_dim=10)
    val_loader = DataLoader(validset, batch_size=32, shuffle=False)

    # Define model and other necessary parameters
    autoencoder = VariationalAutoEncoder(
        spectral_emb_dim + 1, hidden_dim_encoder=64, hidden_dim_decoder=256,
        latent_dim=32, n_layers_encoder=2, n_layers_decoder=3, n_max_nodes=50
    ).to(device)
    denoise_model = DenoiseNN(
        input_dim=32, hidden_dim=512, n_layers=3, n_cond=7, d_cond=128
    ).to(device)

    # Load pretrained weights
    autoencoder.load_state_dict(torch.load('autoencoder.pth.tar')['state_dict'])
    denoise_model.load_state_dict(torch.load('denoise_model.pth.tar')['state_dict'])

    # Define diffusion parameters
    timesteps = 500
    betas = linear_beta_schedule(timesteps)

    # Compute MAE
    mae = compute_mae(denoise_model, val_loader, latent_dim=32, timesteps=timesteps, betas=betas, device=device)
    print(f"Mean Absolute Error (MAE) on validation set: {mae:.4f}")

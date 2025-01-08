import networkx as nx
import random
import numpy as np
import os
import re
import networkx as nx
import numpy as np
import pandas as pd

## GENERATION DE GRAPH

def generate_sbm_graph(n_nodes, n_communities, p_intra, p_inter):
    """Génère un graphe SBM avec les communautés spécifiées."""
    sizes = [n_nodes // n_communities] * n_communities
    probs = [[p_intra if i == j else p_inter for j in range(n_communities)]
             for i in range(n_communities)]
    graph = nx.stochastic_block_model(sizes, probs)
    return graph


def adjust_edges(graph, target_edges, target_average_degree):
    """Ajuste les arêtes pour se rapprocher des cibles."""
    current_edges = graph.number_of_edges()
    while current_edges < target_edges:
        # Ajouter une arête aléatoire
        u, v = random.sample(graph.nodes, 2)
        if not graph.has_edge(u, v):
            graph.add_edge(u, v)
        current_edges += 1
    
    while current_edges > target_edges:
        # Supprimer une arête aléatoire
        u, v = random.choice(list(graph.edges))
        graph.remove_edge(u, v)
        current_edges -= 1
    return graph

def add_triangles(graph, target_triangles):
    """Ajoute des triangles au graphe pour se rapprocher de la cible."""
    current_triangles = sum(nx.triangles(graph).values()) // 3
    while current_triangles < target_triangles:
        # Trouver deux voisins d'un nœud qui ne sont pas connectés
        nodes = list(graph.nodes)
        random.shuffle(nodes)
        for u in nodes:
            neighbors = list(graph.neighbors(u))
            if len(neighbors) < 2:
                continue
            v, w = random.sample(neighbors, 2)
            if not graph.has_edge(v, w):
                graph.add_edge(v, w)
                current_triangles += 1
                break
    return graph


def validate_and_adjust(graph, target_params):
    """Valide et ajuste les paramètres du graphe."""
    current_params = {
        'n_nodes': graph.number_of_nodes(),
        'n_edges': graph.number_of_edges(),
        'average_degree': sum(dict(graph.degree).values()) / graph.number_of_nodes(),
        'n_triangles': sum(nx.triangles(graph).values()) // 3,
        'n_communities': len(list(nx.connected_components(graph)))
    }
    for key, target in target_params.items():
        print(f"{key}: target = {target}, current = {current_params[key]}")
    return graph


def generate_graph(n_nodes, n_edges, average_degree, n_triangles, n_communities):
    """Génère un graphe respectant les paramètres donnés."""
    # Étape 1 : Génération d'un graphe SBM
    p_intra = 0.6  # Probabilité intra-communautés
    p_inter = 0.1  # Probabilité inter-communautés
    graph = generate_sbm_graph(n_nodes, n_communities, p_intra, p_inter)
    
    # Étape 2 : Ajustement du nombre d’arêtes
    graph = adjust_edges(graph, n_edges, average_degree)
    
    # Étape 3 : Ajout de triangles
    graph = add_triangles(graph, n_triangles)
    
    # Étape 4 : Validation et ajustements finaux
    target_params = {
        'n_nodes': n_nodes,
        'n_edges': n_edges,
        'average_degree': average_degree,
        'n_triangles': n_triangles,
        'n_communities': n_communities
    }
    graph = validate_and_adjust(graph, target_params)
    return graph

### EXTRACTION DES STATISTIQUES

def extract_statistics_from_description(file_path):
    """Extrait les statistiques depuis un fichier description en utilisant les positions des nombres."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Trouver tous les nombres dans le texte (entiers ou flottants)
    numbers = re.findall(r'[\d.]+', content)
    
    # S'assurer qu'il y a suffisamment de nombres pour extraire les statistiques
    if len(numbers) < 7:
        raise ValueError(f"Le fichier {file_path} ne contient pas suffisamment de données pour extraire les statistiques.")
    
    # Extraire les statistiques nécessaires
    n_nodes = int(numbers[0])  # 1er nombre
    n_edges = int(numbers[1])  # 2ème nombre
    average_degree = float(numbers[2])  # 3ème nombre
    n_triangles = int(numbers[3])  # 4ème nombre
    n_communities = int(numbers[6])  # 7ème nombre
    
    return [n_nodes, n_edges, average_degree, n_triangles, n_communities]


def extract_statistics_from_graph(graph_path):
    """Charge un graphe depuis un fichier et calcule ses statistiques."""
    graph = nx.read_edgelist(graph_path)
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    average_degree = sum(dict(graph.degree).values()) / n_nodes if n_nodes > 0 else 0
    n_triangles = sum(nx.triangles(graph).values()) // 3
    communities = list(nx.connected_components(graph))  # Connexité pour les communautés
    n_communities = len(communities)
    return [n_nodes, n_edges, average_degree, n_triangles, n_communities]




### PIPELINE

def main():
    # Dossiers et fichiers
    description_dir = "../data/train/description/"
    graph_dir = "../data/valid/graph/"
    
    # Extraction des vraies statistiques
    true_stats = []
    for desc_file in sorted(os.listdir(description_dir)):
        file_path = os.path.join(description_dir, desc_file)
        stats = extract_statistics_from_description(file_path)
        if stats:
            true_stats.append(stats)
        else: print("error")
    
    true_stats = np.array(true_stats)
    mean_stats = true_stats.mean(axis=0)
    std_stats = true_stats.std(axis=0)
    
    # Génération et évaluation
    results = []
    for i, graph_file in enumerate(sorted(os.listdir(graph_dir))):
        graph_path = os.path.join(graph_dir, graph_file)
        
        # Statistiques vraies
        true_stats_graph = extract_statistics_from_graph(graph_path)
        
        # Génération d'un graphe
        generated_graph = generate_graph(*true_stats_graph)
        
        # Statistiques générées
        generated_stats = extract_statistics_from_graph(graph_path)
        
        # Normalisation
        true_stats_normalized = (np.array(true_stats_graph)- mean_stats)/ std_stats
        generated_stats_normalized = (np.array(generated_stats)- mean_stats)/ std_stats
        
        # Enregistrement des résultats
        results.append({
            "graph_id": i,
            "true_stats": true_stats_normalized.tolist(),
            "generated_stats": generated_stats_normalized.tolist()
        })
    
    # Calcul du score MAE
    mae = np.mean([np.abs(np.array(r["true_stats"]) - np.array(r["generated_stats"])).mean() for r in results])
    print(f"Mean Absolute Error (MAE): {mae}")
    
    # Sauvegarde des résultats
    #results_df = pd.DataFrame(results)
    #results_df.to_csv("results.csv", index=False)


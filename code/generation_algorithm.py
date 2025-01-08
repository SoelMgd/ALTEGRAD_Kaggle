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

def add_triangles(graph, target_triangles, max_attempts=50):
    """
    Ajoute des triangles au graphe pour se rapprocher de la cible.
    Limite le nombre d'itérations pour éviter les boucles infinies.
    
    Parameters:
        graph (nx.Graph): Le graphe à modifier.
        target_triangles (int): Nombre cible de triangles.
        max_attempts (int): Nombre maximum de tentatives pour ajouter des triangles.
    """
    current_triangles = sum(nx.triangles(graph).values()) // 3
    attempts = 0

    while current_triangles < target_triangles and attempts < max_attempts:
        attempts += 1
        
        # Trouver deux voisins d'un nœud qui ne sont pas connectés
        nodes = list(graph.nodes)
        random.shuffle(nodes)  # Mélanger les nœuds pour diversifier les tentatives
        
        triangle_added = False
        for u in nodes:
            neighbors = list(graph.neighbors(u))
            if len(neighbors) < 2:
                continue
            v, w = random.sample(neighbors, 2)
            if not graph.has_edge(v, w):
                graph.add_edge(v, w)  # Ajouter l'arête pour former un triangle
                current_triangles += 1
                triangle_added = True
                break
        
        # Si aucun triangle n'a pu être ajouté, on arrête
        if not triangle_added:
            break

    if attempts == max_attempts:
        print(f"[WARNING] Reached max attempts ({max_attempts}) while adding triangles.")
    
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
    #for key, target in target_params.items():
        #print(f"{key}: target = {target}, current = {current_params[key]}")
    return graph


def generate_graph(n_nodes, n_edges, average_degree, n_triangles, n_communities):
    """Génère un graphe respectant les paramètres donnés."""
    # Étape 1 : Génération d'un graphe SBM
    p_intra = 0.6  # Probabilité intra-communautés
    p_inter = 0.1  # Probabilité inter-communautés
    print("[--INFO] Génération du graph")
    graph = generate_sbm_graph(n_nodes, n_communities, p_intra, p_inter)
    
    # Étape 2 : Ajustement du nombre d’arêtes
    print("[--INFO] Ajustement du nombre d'arrêtes")
    graph = adjust_edges(graph, n_edges, average_degree)
    
    # Étape 3 : Ajout de triangles
    print("[--INFO] Ajustement du nombre de triangles")
    graph = add_triangles(graph, n_triangles)
    
    print("[--INFO] Validation et ajustement")
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

def extract_numbers(text):
    # Use regular expression to find integers and floats
    numbers = re.findall(r'\d+\.\d+|\d+', text)
    # Convert the extracted numbers to float
    return [float(num) for num in numbers]


def extract_feats(file):
    stats = []
    fread = open(file,"r")
    line = fread.read()
    line = line.strip()
    stats = extract_numbers(line)
    fread.close()
    return stats

def extract_statistics_from_description(file):
    """
    Extrait les statistiques depuis un fichier description en utilisant des fonctions fiables.
    """
    # Extraire les nombres depuis le texte
    stats = extract_feats(file)

    # Vérifier qu'il y a assez de valeurs
    if len(stats) < 7:
        raise ValueError(f"Le fichier {file} ne contient pas suffisamment de données pour extraire les statistiques.")
    
    # Sélectionner les statistiques nécessaires
    n_nodes = int(stats[0])  # 1er nombre
    n_edges = int(stats[1])  # 2ème nombre
    average_degree = float(stats[2])  # 3ème nombre
    n_triangles = int(stats[3])  # 4ème nombre
    n_communities = int(stats[6])  # 7ème nombre
    clustering = int(stats[4])
    kcore = int(stats[5])

    return [n_nodes, n_edges, average_degree, n_triangles, n_communities, clustering, kcore]

def custom_read_edgelist(file_path):
    """
    Lit un fichier edgelist et ignore les métadonnées inutiles `{}`.
    
    Parameters:
        file_path (str): Chemin vers le fichier edgelist.
    
    Returns:
        nx.Graph: Graphe NetworkX construit à partir des arêtes valides.
    """
    edges = []
    with open(file_path, 'r') as f:
        for line in f:
            # Ignorer les lignes vides ou mal formatées
            if line.strip():
                try:
                    # Extraire uniquement les deux premiers éléments (nœuds)
                    u, v = line.strip().split()[:2]
                    edges.append((u, v))
                except ValueError:
                    # Si une ligne n'a pas exactement 2 ou plus éléments, l'ignorer
                    continue
    
    # Construire un graphe à partir des arêtes valides
    graph = nx.Graph()
    graph.add_edges_from(edges)
    return graph


def extract_statistics_from_graph(graph_input):
    """
    Extrait les statistiques d'un graphe à partir d'un chemin ou d'un objet NetworkX.
    
    Parameters:
        graph_input (str ou nx.Graph): Chemin vers le fichier ou graphe NetworkX.
    
    Returns:
        list: [n_nodes, n_edges, average_degree, n_triangles, n_communities, clustering, kcore]
    """
    if isinstance(graph_input, str):  # Si c'est un chemin vers un fichier
        graph = custom_read_edgelist(graph_input)  # Utilise la fonction personnalisée
    elif isinstance(graph_input, nx.Graph):  # Si c'est un objet NetworkX
        graph = graph_input
    else:
        raise ValueError("L'entrée doit être un chemin vers un fichier ou un objet NetworkX.")
    
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    average_degree = sum(dict(graph.degree).values()) / n_nodes if n_nodes > 0 else 0
    n_triangles = sum(nx.triangles(graph).values()) // 3
    n_communities = len(list(nx.connected_components(graph)))
    clustering = nx.transitivity(graph)
    kcore = max(nx.core_number(graph).values())
    
    return [n_nodes, n_edges, average_degree, n_triangles, n_communities, clustering, kcore]





### PIPELINE

def pipe_valid():
    # Dossiers et fichiers
    description_dir = "../data/train/description/"
    graph_dir = "../data/valid/graph/"
    
    # Extraction des vraies statistiques
    print("[INFO] Extraction des statistiques")
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
    print("[INFO] Génération")
    results = []
    limit = np.inf
    for i, graph_file in enumerate(sorted(os.listdir(graph_dir))):
        if i >= limit:
            break 
        
        graph_path = os.path.join(graph_dir, graph_file)
        
        # Statistiques vraies
        true_stats_graph = extract_statistics_from_graph(graph_path)
        
        # Génération d'un graphe
        generated_graph = generate_graph(*true_stats_graph[:5])
        
        # Statistiques générées
        generated_stats = extract_statistics_from_graph(generated_graph)
        
        # Normalisation
        true_stats_normalized = (np.array(true_stats_graph)- mean_stats)/ std_stats
        generated_stats_normalized = (np.array(generated_stats)- mean_stats)/ std_stats
        
        # Enregistrement des résultats
        results.append({
            "graph_id": i,
            "true_stats": true_stats_normalized.tolist(),
            "generated_stats": generated_stats_normalized.tolist()
        })
        print("Graph ", i, results[-1])

    # Calcul du score MAE
    mae = np.mean([np.abs(np.array(r["true_stats"]) - np.array(r["generated_stats"])).mean() for r in results])
    print(f"Mean Absolute Error (MAE): {mae}")
    
    # Sauvegarde des résultats
    #results_df = pd.DataFrame(results)
    #results_df.to_csv("results.csv", index=False)


pipe_valid()


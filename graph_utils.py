import jax
import jax.numpy as jnp
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def generate_er_graph_jax(n, p, seed=None):
    """
    Takes:
    n: int
        Number of nodes
    p: float
        Probability of edge
    seed: jax.random.PRNG seed
        Seed for reproducibility
    Returns:
    Erdős-Rényi random graph converted to a JAX array adjacency matrix.
    """

    graph = nx.erdos_renyi_graph(n, p, seed=seed)

    adjacency_matrix = jnp.array(nx.to_numpy_array(graph))
    return adjacency_matrix

def visualize_graph(adjacency_matrix):
    
    G = nx.from_numpy_array(adjacency_matrix)

    plt.figure(figsize=(6, 6))
    nx.draw_networkx(G, with_labels=False, node_color='lightgreen', 
                    node_size=100, edge_color='black', width=0.3)
    plt.axis('off')
    plt.show()
import jax
import jax.numpy as jnp
import networkx as nx

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

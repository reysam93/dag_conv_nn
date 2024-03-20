import numpy as np
from numpy import linalg as la
import networkx as nx


def create_dag(N, p, weighted=True):
    """
    Create a random directed acyclic graph (DAG) with independent edge probability.

    Args:
        N (int): Number of nodes.
        p (float): Probability of edge creation.
        weighted (bool, optional): Whether to generate a weighted DAG. Defaults to True.

    Returns:
        tuple[np.ndarray, nx.DiGraph]: Tuple containing the adjacency matrix and the DAG.
    """
    graph = nx.erdos_renyi_graph(N, p, directed=True)
    Adj = nx.to_numpy_array(graph)
    Adj = np.tril(Adj, k=-1) 

    if weighted:
        Weights = np.random.uniform(low=0.2, high=1, size=(N, N))
        Adj = Adj * Weights
        colums_sum = Adj.sum(axis=0)
        col_sums_nonzero = colums_sum[colums_sum != 0]
        Adj[:, colums_sum != 0] /= col_sums_nonzero

    dag = nx.from_numpy_array(Adj.T, create_using=nx.DiGraph())
    return Adj, dag


def compute_Dq(dag: nx.DiGraph, target_node: str, only_diag: bool = True,
               verbose: bool = False) -> np.ndarray:
    """
    Compute Dq, the frequency response matrix of the GSO associated with node q, based on the
    existence of paths from each node to the target node.

    Args:
        dag (nx.DiGraph): Directed acyclic graph (DAG).
        target_node (str): Target node identifier.
        only_diag (bool, optional): Whether to return only the diagonal of the matrix. Defaults to True.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        np.ndarray: Frequency response matrix Dq.
    """
    N = dag.number_of_nodes()
    target_idx = ord(target_node) - ord('a') if isinstance(target_node, str) else target_node
    
    path_exists = np.zeros(N)
    for i in range(target_idx+1):
        path_exists[i] = nx.has_path(dag, i, target_idx)
    
    if verbose:
        for i, exists in enumerate(path_exists):
            print(f'Has path from node {i} to node {target_node}: {exists}')

    if only_diag:
        return path_exists
    else:
        return np.diag(path_exists)


def compute_GSOs(W, dag):
    N = W.shape[0]
    GSOs = np.array([W @ compute_Dq(dag, i, N) @ la.inv(W) for i in range(N)])
    return GSOs

import numpy as np
from numpy import linalg as la
import networkx as nx
from torch import Tensor


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
    assert nx.is_directed_acyclic_graph(dag), "Graph is not a DAG"
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

def create_DAG_fitler(GSOs, norm_coefs=False, ftype='uniform'):
    """
    Create a directed acyclic graph (DAG) filter based on the provided graph shift operators (GSOs).

    Args:
    - GSOs: ndarray, shape (K, N, N), where K is the number of GSOs and N is the number of nodes.
    - norm_coefs: bool, whether to normalize the filter coefficients.
    - ftype: str, the type of filter coefficients to generate. Options: 'uniform', 'normal'.

    Returns:
    - H: ndarray, shape (N, N), the constructed DAG filter.
    - filt_coefs: ndarray, shape (K,), the filter coefficients used.
    """
    # Select GSOs and create GF
    if ftype == 'uniform':
        filt_coefs = 2*np.random.rand(GSOs.shape[0]) - 1
    elif type == 'uniform-pos':
        filt_coefs = np.random.rand(GSOs.shape[0]) + .1
    else:
        filt_coefs = np.random.randn(GSOs.shape[0])
    
    if norm_coefs:
        filt_coefs /= la.norm(filt_coefs, 1)

    H = (filt_coefs[:, None, None] * GSOs).sum(axis=0)
    return H, filt_coefs 

def create_diff_data(M, GSOs, max_src_node, n_p=.1, n_sources=1, norm_y='l2_norm',
                     norm_f_coefs=False, src_t='constant', ftype='uniform', torch_tensor=False,
                     mask_sources=False):    
    """
    Create data following a diffusion proces that is modeled via a graph filter
    for DAGs.

    Args:
    - M: int, number of samples to generate.
    - GSOs: ndarray, shape (K, N, N), where K is the number of GSOs and N is the number of nodes.
    - max_src_node: int, maximum source node index for generating sparse input signals.
    - n_p: float, standard deviation of noise to add to the output signals. Default is 0.1.
    - n_sources: int, number of sources to activate in each sparse input signal. Default is 1.
    - norm_y: str, method for normalizing the output signals Y. Options: 'l2_norm', 'standardize', or None. Default is 'l2_norm'.
    - norm_f_coefs: bool, whether to normalize the filter coefficients used to generate the output signals.
    - ftype: str, type of filter coefficients. Options: 'uniform', 'normal'. Default is 'uniform'.

    Returns:
    - Y: ndarray, shape (M, N, 1), the generated output signals.
    - X: ndarray, shape (M, N, 1), the generated sparse input signals.
    """
    assert (max_src_node >= n_sources), 'Number of sources must be smaller than maximum source node or random'



    # Generate sparse input signals
    N = GSOs.shape[1]
    X = np.zeros((M, N))
    idx = np.random.randint(0, max_src_node, (M, n_sources))
    row_idx = np.arange(M).reshape(-1, 1)

    # Create random non-zero values
    if src_t == 'random':
        pos_samples = np.random.uniform(.5, 1.5, int(n_sources*M/2))
        neg_samples = np.random.uniform(-1.5, -.5, int(n_sources*M/2))
        all_samples = np.concatenate((pos_samples, neg_samples))
        np.random.shuffle(all_samples)
        values = all_samples.reshape([M, n_sources])
    else:
        values = 1  / np.sqrt(n_sources)
    X[row_idx, idx[:]] = values

    H, _ = create_DAG_fitler(GSOs, norm_f_coefs, ftype='uniform')

    # Generate output signals
    Y = X @ H.T

    # Normalize output signals if required
    signal_norm = la.norm(Y, 2, axis=1, keepdims=True)
    if norm_y == 'l2_norm':
        Y = Y / signal_norm
    elif norm_y == 'standarize':
        Y = (Y - np.mean(Y, axis=1, keepdims=True)) / np.std(Y, axis=1, keepdims=True)

    # Add noise
    if n_p > 0:
        noise = np.random.randn(M, N)
        noise_norm = la.norm(noise, 2, axis=1, keepdims=True)
        noise = noise * signal_norm * n_p / noise_norm
        Y = Y + noise         

    # Mask output values related to source nodes to avoid learning trivial mapping
    if mask_sources:
        mask = np.ones_like(Y)
        mask[:,:max_src_node] = 0
        Y = Y*mask

    Y, X = np.expand_dims(Y, axis=2), np.expand_dims(X, axis=2)
    if torch_tensor:
        return Tensor(Y), Tensor(X)
    else:
        return Y, X

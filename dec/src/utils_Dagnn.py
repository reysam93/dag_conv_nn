import os
import numpy as np
from numpy import linalg as la
import networkx as nx
import torch
from torch import Tensor
from torch_geometric.data import Data

import igraph as ig


path = os.path.abspath(__file__)
path = path[:path.rindex("/")] + "/../"
PATH = os.path.abspath(path)


DIR_DATA = os.path.join(PATH, 'data')
DIR_RESULTS = os.path.join(PATH, 'results')
DIR_SAVED_MODELS = os.path.join(PATH, 'saved_models')

NA_SUM = "add"
NA_MAX = "max"
NA_GATED_SUM = "gated_sum"
NA_SELF_ATTN_X = "self_attn_x"  # use xs of preds to compute weights, used to aggregate hs of preds
NA_SELF_ATTN_H = "self_attn_h"
NA_ATTN_X = "attn_x"  # use x and xs of preds to compute weights, used to aggregate hs of preds
NA_ATTN_H = "attn_h"  # use x and hs of preds
NA_MATTN_H = "mattn_h"  # use x and hs of preds

P_MEAN = "mean"
P_ADD = "add"  # do not use "sum" so that can be used to call tg pooling function
P_SUM = "sum"  # do not use "sum" so that can be used to call tg pooling function
P_MAX = "max"
P_ATTN = "attn"
EMB_POOLINGS = [P_MEAN, P_MAX, P_SUM]
POOLINGS = [P_MEAN, P_MAX, P_ATTN, P_ADD]



# see https://github.com/unbounce/pytorch-tree-lstm/blob/66f29a44e98c7332661b57d22501107bcb193f90/treelstm/util.py#L8
# assume nodes consecutively named starting at 0
#
def top_sort(edge_index, graph_size):

    node_ids = np.arange(graph_size, dtype=int)

    node_order = np.zeros(graph_size, dtype=int)
    unevaluated_nodes = np.ones(graph_size, dtype=bool)

    parent_nodes = edge_index[0]
    child_nodes = edge_index[1]

    n = 0
    while unevaluated_nodes.any():
        # Find which parent nodes have not been evaluated
        unevaluated_mask = unevaluated_nodes[parent_nodes]

        # Find the child nodes of unevaluated parents
        unready_children = child_nodes[unevaluated_mask]

        # Mark nodes that have not yet been evaluated
        # and which are not in the list of children with unevaluated parent nodes
        nodes_to_evaluate = unevaluated_nodes & ~np.isin(node_ids, unready_children)

        node_order[nodes_to_evaluate] = n
        unevaluated_nodes[nodes_to_evaluate] = False

        n += 1

    return torch.from_numpy(node_order).long()


def add_order_info_01(graph):

    l0 = top_sort(graph.edge_index, graph.num_nodes)
    ei2 = torch.LongTensor([list(graph.edge_index[1]), list(graph.edge_index[0])])
    l1 = top_sort(ei2, graph.num_nodes)
    ns = torch.LongTensor([i for i in range(graph.num_nodes)])

    graph.__setattr__("_bi_layer_idx0", l0)
    graph.__setattr__("_bi_layer_index0", ns)
    graph.__setattr__("_bi_layer_idx1", l1)
    graph.__setattr__("_bi_layer_index1", ns)

    assert_order(graph.edge_index, l0, ns)
    assert_order(ei2, l1, ns)


def assert_order(edge_index, o, ns):
    # already processed
    proc = []
    for i in range(max(o)+1):
        # nodes in position i in order
        l = o == i
        l = ns[l].tolist()
        for n in l:
            # predecessors
            ps = edge_index[0][edge_index[1] == n].tolist()
            for p in ps:
                assert p in proc
        proc += l


def add_order_info(graph):
    ns = torch.LongTensor([i for i in range(graph.num_nodes)])
    layers = torch.stack([top_sort(graph.edge_index, graph.num_nodes), ns], dim=0)
    ei2 = torch.LongTensor([list(graph.edge_index[1]), list(graph.edge_index[0])])
    layers2 = torch.stack([top_sort(ei2, graph.num_nodes), ns], dim=0)

    graph.__setattr__("bi_layer_index", torch.stack([layers, layers2], dim=0))



def add_noise(signal, n_p):
    shape = signal.shape

    M = shape[0]
    N = shape[1]
 
    if n_p <= 0:
        return signal
    
    if not isinstance(signal, torch.Tensor):
        signal = torch.tensor(signal, dtype=torch.float32)
    signal_norm = torch.norm(signal, p=2, dim=1, keepdim=True)

    signal_norm[signal_norm == 0] = 1
    noise = torch.randn(M, N, 1, device=signal.device)
    noise_norm = torch.norm(noise, p=2, dim=1, keepdim=True)
    noise = noise * signal_norm * torch.sqrt(torch.tensor(n_p)) / noise_norm
    
    return signal + noise

def create_dag(N, p, weighted=False, weakly_conn=True, max_tries=25):
    """
    Create a random directed acyclic graph (DAG) with independent edge probability.

    Args:
        N (int): Number of nodes.
        p (float): Probability of edge creation.
        weighted (bool, optional): Whether to generate a weighted DAG. Defaults to True.

    Returns:
        tuple[np.ndarray, nx.DiGraph]: Tuple containing the adjacency matrix and the DAG.
    """
    for _ in range (max_tries):
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

        if not weakly_conn or nx.is_weakly_connected(dag):
            assert nx.is_directed_acyclic_graph(dag), "Graph is not a DAG"
            return Adj, dag
    
    print('WARING: dag is not weakly connected')
    assert nx.is_directed_acyclic_graph(dag), "Graph is not a DAG"
    return Adj, dag

def compute_Dq(dag: nx.DiGraph, target_node: str, only_diag: bool = True,
               verbose: bool = False, ordered: bool = False) -> np.ndarray:
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
    max_node = target_idx + 1 if ordered else N 
    for i in range(max_node):
        path_exists[i] = nx.has_path(dag, i, target_idx)
    
    if verbose:
        for i, exists in enumerate(path_exists):
            print(f'Has path from node {i} to node {target_node}: {exists}')

    if only_diag:
        return path_exists
    else:
        return np.diag(path_exists)

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


def add_noise1(signal, n_p):
    M, N = signal.shape

    if n_p <= 0:
        return signal
    
    signal_norm = la.norm(signal, 2, axis=1, keepdims=True)
    signal_norm[signal_norm == 0] = 1

    noise = np.random.randn(M, N)
    noise_norm = la.norm(noise, 2, axis=1, keepdims=True)
    noise = noise * signal_norm * np.sqrt(n_p) / noise_norm
    return signal + noise


    
def create_diff_data(M, GSOs, max_src_node, n_p_x=0, n_p_y=0, n_sources=1, norm_y='l2_norm',
                     norm_f_coefs=False, src_t='constant', ftype='uniform', torch_tensor=False,
                     mask_sources=False, verb=False):    
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

    H, _ = create_DAG_fitler(GSOs, norm_f_coefs, ftype=ftype)

    # Generate output signals
    Y = X @ H.T

    # Normalize output signals if required
    if norm_y == 'l2_norm':
        signal_norm = la.norm(Y, 2, axis=1, keepdims=True)
        signal_norm[signal_norm == 0] = 1
        Y = Y / signal_norm
    elif norm_y == 'standarize':
        Y = (Y - np.mean(Y, axis=1, keepdims=True)) / np.std(Y, axis=1, keepdims=True)

    Xn = add_noise1(X, n_p_x)
    Yn = add_noise1(Y, n_p_y)
   
    if verb:
        noise_x_err = (la.norm(Xn - X, 2, axis=1)**2).mean()
        noise_y_err = (la.norm(Yn - Y, 2, axis=1)**2).mean()
        print(f'Noise power for X: {n_p_x} --> |Xn - X|^2={noise_x_err:.4f}')
        print(f'Noise power for Y: {n_p_y} --> |Yn - Y|^2={noise_y_err:.4f}')

    if mask_sources:
        mask = np.ones_like(Y)
        mask[:,:max_src_node] = 0
        Y = Y*mask
        Yn = Yn*mask

    Yn = np.expand_dims(Yn, axis=2)
    Y = np.expand_dims(Y, axis=2)
    Xn = np.expand_dims(Xn, axis=len(Xn.shape))
    if torch_tensor:
        return Tensor(Yn), Tensor(Xn), Tensor(Y)
    else:
        return Yn, Xn, Y





def get_signals(d_p, GSOs):
    range_GSO = np.arange(d_p['min_GSO'], d_p['max_GSO'])
    gsos_idx = np.random.choice(range_GSO, size=d_p['n_GSOs'], replace=False)
    sel_GSOs = GSOs[gsos_idx]
    Yn_t, X_t, Y_t = create_diff_data(d_p['M'], sel_GSOs, d_p['max_src_node'], d_p['n_p_x'], d_p['n_p_y'],
                                           d_p['n_sources'], src_t=d_p['src_t'], torch_tensor=True, verb=False)
    # print(Yn_t.shape)
    # X_data = {'train': X_t[:d_p['M_train']], 'val': X_t[d_p['M_train']:-d_p['M_test']], 'test': X_t[-d_p['M_test']:]}
    # Y_data = {'train': Yn_t[:d_p['M_train']], 'val': Yn_t[d_p['M_train']:-d_p['M_test']],
    #           'test': Y_t[-d_p['M_test']:]}
        
    return Yn_t, X_t, Y_t


def get_graph_data(d_dat_p, get_Psi=False):

    Adj, dag = create_dag(d_dat_p['N'], d_dat_p['p'])
    W = la.inv(np.eye(d_dat_p['N']) - Adj)
    W_inf = la.inv(W)

    if get_Psi:
        Psi = np.array([compute_Dq(dag, i, d_dat_p['N']) for i in range(d_dat_p['N'])]).T
        GSOs = np.array([(W * Psi[:,i]) @ W_inf for i in range(d_dat_p['N'])])
        return Adj, W, GSOs, Psi
    
    GSOs = np.array([(W * compute_Dq(dag, i, d_dat_p['N'])) @ W_inf for i in range(d_dat_p['N'])])

    return Adj, W, GSOs





def DAGNN_model(adj, X_dict, Y_dict):
    X_processed = {}
    Y_processed = {}

    adj = adj.T

    for label in ['train', 'val', 'test']:
        graph_signal_pairs = []

        X = X_dict[label]
        Y = Y_dict[label]

        for i in range(len(Y)):
            g = nx.DiGraph(adj)
            edge_index = torch.tensor(list(g.edges)).t().contiguous()
            x = X[i].float()
            graph_data = Data(x=x, edge_index=edge_index)
            add_order_info(graph_data)
            graph_data.vs = [{'type': 0}] * len(g.nodes)
            graph_data.y = Y[i]
            
            graph_signal_pairs.append(graph_data)

        X_processed[label] = graph_signal_pairs
        Y_processed[label] = Y

    return X_processed, Y_processed


def DVAE_model(adj, X_dict, Y_dict):
    X_processed = {}
    Y_processed = {}

    adj = adj.T

    for label in ['train', 'val', 'test']:
        graph_signal_pairs = []

        X = X_dict[label] 
        Y = Y_dict[label]

        for i in range(len(Y)):
            g = ig.Graph(directed=True)
            g.add_vertices(adj.shape[0])  
            # edges = []
            # for i in range(adj.shape[0]-1):
            #     edges.append((i, i+1))

            edges = []
            for row in range(adj.shape[0]):
                for col in range(adj.shape[0]):
                    if adj[row,col] != 0:  # Check for any non-zero weight
                        edges.append((row,col))
            g.add_edges(edges)
            for v in g.vs:
                idx = v.index
                v['scalar_value'] = X[i, idx, 0].item()

            g['y'] = Y[i]
            graph_signal_pairs.append(g)

        X_processed[label] = graph_signal_pairs
        Y_processed[label] = Y

    return X_processed, Y_processed


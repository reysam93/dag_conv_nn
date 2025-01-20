import networkx as nx
from torch import Tensor
import torch
import numpy as np
from numpy import linalg as la
import dgl
import matplotlib.pyplot as plt
from pandas import DataFrame
from IPython.display import display

from src.arch import DAGConv, FB_DAGConv, ADCN, ParallelMLPSum, SharedMLPSum
from src.baselines_archs import GAT, MLP, MyGCNN, GraphSAGE, GIN
import src.dag_utils as dagu
from src.DAGNN import DVAE, DAGNN
# from src.baselines.dagnn import DAGNN
from torch_geometric.data import Data


def get_graph_data(d_dat_p, get_Psi=False):
    Adj, dag = dagu.create_dag(d_dat_p['N'], d_dat_p['p'])
    W = la.inv(np.eye(d_dat_p['N']) - Adj)
    W_inv = la.inv(W)

    if get_Psi:
        Psi = np.array([dagu.compute_Dq(dag, i) for i in range(d_dat_p['N'])]).T
        GSOs = np.array([(W * Psi[:,i]) @ W_inv for i in range(d_dat_p['N'])])
        return Adj, W, GSOs, Psi
    
    GSOs = np.array([(W * dagu.compute_Dq(dag, i)) @ W_inv for i in range(d_dat_p['N'])])
    return Adj, W, GSOs

def kipf_GSO(S):
    S_hat = S + np.eye(S.shape[0])
    Deg_sqrt_inv = np.diag(1/np.sqrt(S_hat.sum(axis=0)))
    return Deg_sqrt_inv @ S_hat @ Deg_sqrt_inv


def select_GSO(arc_p, GSOs, sel_GSOs, W, Adj, sel_GSOs_idx=None):
    transp = 'transp' in arc_p.keys() and arc_p['transp']
    transp_GSO = lambda GSO, transp: np.transpose(GSO, axes=[0,2,1]) if transp else GSO

    # Original GSOs
    if arc_p['GSO'] == 'GSOs':
        return Tensor(transp_GSO(GSOs, transp))
    elif arc_p['GSO'] == 'sel_GSOs':
        return Tensor(transp_GSO(sel_GSOs, transp))
    elif arc_p['GSO'] == 'rnd_GSOs':
        rnd_idx = np.random.choice(GSOs.shape[0], size=arc_p['n_gsos'], replace=False)
        return  Tensor(transp_GSO(GSOs[rnd_idx], transp))
    elif arc_p['GSO'] == 'no_sel_GSOs':
        non_sel_idx = np.setdiff1d(np.arange(GSOs.shape[0]), sel_GSOs_idx)
        replace = len(non_sel_idx) > GSOs.shape[0]
        rnd_idx = np.random.choice(non_sel_idx, size=arc_p['n_gsos'], replace=replace)
        return  Tensor(transp_GSO(GSOs[rnd_idx], transp))
    elif arc_p['GSO'] == 'first_GSOs':
        return Tensor(transp_GSO(GSOs[:arc_p['n_gsos']], transp))
    elif arc_p['GSO'] == 'last_GSOs':
        return Tensor(transp_GSO(GSOs[-arc_p['n_gsos']:], transp))
    elif arc_p['GSO'] == 'A_pows':
        A_pows = np.array([np.linalg.matrix_power(Adj, k) for k in range(arc_p['K'])])
        return Tensor(transp_GSO(A_pows, transp))
    elif arc_p['GSO'] == 'W-dgl':
        W_aux = W.T if transp else W
        return dgl.from_networkx(nx.from_numpy_array(W_aux)).add_self_loop()
    elif arc_p['GSO'] == 'A-dgl':
        Adj_aux = Adj.T if transp else Adj
        return dgl.from_networkx(nx.from_numpy_array(Adj_aux)).add_self_loop()
    elif arc_p['GSO'] == 'W':
        W_aux = W.T if transp else W
        return Tensor(kipf_GSO(W_aux))
    elif arc_p['GSO'] == 'A':
        Adj_aux = Adj.T if transp else Adj
        return Tensor(kipf_GSO(Adj_aux))
    elif arc_p['GSO'] == 'A-pyg':
        # For DAGNN
        Adj_aux = Adj.T if transp else Adj
        row, col = np.nonzero(Adj)
        edge_index = torch.tensor(np.vstack((row, col)), dtype=torch.long)
        graph = Data(edge_index=edge_index, num_nodes=Adj_aux.shape[0])
        add_order_info_01(graph)
        return graph
    else:
        return None
    


def instantiate_arch(arc_p, K):
    args = arc_p['args']
    if arc_p['arch'] in [DAGConv, FB_DAGConv]:
        args['K'] = K

    elif arc_p['arch'] == GAT:
        args = {k: v for k, v in args.items() if k not in ['bias', 'n_layers']}

    elif arc_p['arch'] == DVAE:
        args1 = {}
        args1['max_n'] = arc_p['max_n'] 
        args1['nvt'] = arc_p['nvt'] 
        args1['START_TYPE'] = arc_p['START_TYPE'] 
        args1['END_TYPE'] = arc_p['END_TYPE'] 
        args1['hs'] = arc_p['hs'] 
        args1['nz'] = arc_p['nz'] 
        args1['bidirectional'] = arc_p['bidirectional'] 
        args1['vid'] = arc_p['vid'] 

        # args1['hs'] = arc_p['hs']
        # args1['bidirectional'] = arc_p['bidirectional'] 
        return arc_p['arch'](**args1)



    elif arc_p['arch'] == DAGNN:
        args2 = {}
        args2['emb_dim'] = arc_p['emb_dim'] 
        args2['hidden_dim'] = arc_p['hidden_dim'] 
        args2['out_dim'] = arc_p['out_dim'] 
        args2['max_n'] = arc_p['max_n'] 
        args2['nvt'] = arc_p['nvt'] 
        args2['START_TYPE'] = arc_p['START_TYPE'] 
        args2['END_TYPE'] = arc_p['END_TYPE'] 
        args2['hs'] = arc_p['hs'] 
        args2['nz'] = arc_p['nz'] 
        args2['num_layers'] = arc_p['num_layers'] 

        args2['bidirectional'] = arc_p['bidirectional'] 
        args2['agg'] = arc_p['agg'] 
        args2['out_wx'] = arc_p['out_wx'] 
        args2['out_pool_all'] = arc_p['out_pool_all'] 
        args2['out_pool'] = arc_p['out_pool'] 
        args2['dropout'] = arc_p['dropout'] 
        args2['num_nodes'] = arc_p['num_nodes'] 

        return arc_p['arch'](**args2)


    
    elif arc_p['arch'] in [ParallelMLPSum, SharedMLPSum]:
        args3 = {}
        args3['n_inputs'] = arc_p['n_inputs'] 
        args3['input_dim'] = arc_p['input_dim'] 
        args3['hidden_dims'] = arc_p['hidden_dims'] 
        args3['output_dim'] = arc_p['output_dim'] 

        return arc_p['arch'](**args3)
    


    elif arc_p['arch'] in [SMLP] :

        args4 = {}
        args4['in_dim'] = arc_p['in_dim']
        args4['hid_dim'] = arc_p['hid_dim']
        args4['out_dim'] = arc_p['out_dim']
        args4['bias'] = arc_p['bias']


        return arc_p['arch'](**args4)


    elif arc_p['arch'] not in [ADCN, MyGCNN, MLP, GraphSAGE, GIN]:
        raise ValueError('Unknown architecture type')

    return arc_p['arch'](**args)
    

def display_data(exps_leg, err, std, time, metric_label='Err'):
    data = {
        'Exp': exps_leg,
        f'Mean {metric_label}': err.mean(axis=0),
        f'Median {metric_label}': np.median(err, axis=0),
        # 'Mean Std': std.mean(axis=0) if std is not None else None,
        'Mean Std': std.mean(axis=0) if len(std.shape) == 2 else std,
        'time': time.mean(axis=0)
    }
    df = DataFrame(data)
    display(df)


def plot_results(err, x_values, exps, xlabel, ylabel='Mean Err', figsize=(8,5), skip_idx=[],
                 logy=True, plot_fn=plt.plot, n_cols=3, ylim_bottom=1e-2, ylim_top=1, std=None,
                 alpha=.3, prctile_up=None, prctile_low=None):
    plt.figure(figsize=figsize)

    for i, exp in enumerate(exps):
        if i in skip_idx:
            continue
        if logy and plot_fn is plt.plot:
            plot_fn = plt.semilogy

        plot_fn(x_values, err[:,i], exp['fmt'], label=exp['leg'], linewidth=2.0)

        if std is not None:
            up_ci = np.minimum(err[:,i] + std[:,i], 1)
            low_ci = np.maximum(err[:,i] - std[:,i], 0)
            plt.fill_between(x_values, low_ci, up_ci, alpha=alpha)

        if prctile_up is not None and prctile_low is not None:
            plt.fill_between(x_values, prctile_low[:,i], prctile_up[:,i], alpha=alpha)

    plt.ylim(ylim_bottom, ylim_top)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(ncol=n_cols)
    plt.grid(True)
    plt.show()

def load_data(file_name, src_id=False):
    data = np.load(file_name, allow_pickle=True)
    times = data['times']
    Exps = data['exp']
    xvals = data['xvals'] if 'xvals' in data.keys() else None
    if not src_id:
        err = data['err']
        std = data['std']
        return err, std, times, Exps, xvals
    else:
        acc = data['acc']
        return acc, times, Exps, xvals
    
def data_to_csv(fname, models, xaxis, error):
    header = ''
    data = error
    
    data = np.concatenate((xaxis.reshape([xaxis.size, 1]), error), axis=1)

    header = 'xaxis; '  

    for i, model in enumerate(models):
        header += model['leg']
        if i < len(models)-1:
            header += '; '

    np.savetxt(fname, data, delimiter=';', header=header, comments='')
    print('SAVED as:', fname)


# Functions to prepare graphs for DAGNN
def top_sort(edge_index, graph_size):
    node_ids = np.arange(graph_size, dtype=int)

    node_order = np.zeros(graph_size, dtype=int)
    unevaluated_nodes = np.ones(graph_size, dtype=bool)

    parent_nodes = edge_index[0]
    child_nodes = edge_index[1]

    print(parent_nodes)

    n = 0
    while unevaluated_nodes.any():
        # Find which parent nodes have not been evaluated
        unevaluated_mask = unevaluated_nodes[parent_nodes]
        if(unevaluated_mask.shape==()):
            unevaluated_mask=np.array([unevaluated_mask])
        # Find the child nodes of unevaluated parents
        unready_children = child_nodes[unevaluated_mask]

        # Mark nodes that have not yet been evaluated
        # and which are not in the list of children with unevaluated parent nodes
        nodes_to_evaluate = unevaluated_nodes & ~np.isin(node_ids, unready_children)

        node_order[nodes_to_evaluate] = n
        unevaluated_nodes[nodes_to_evaluate] = False

        n += 1

    return torch.from_numpy(node_order).long()
    

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


def add_order_info_01(graph):
    l0 = top_sort(graph.edge_index, graph.num_nodes)
    ei2 = torch.LongTensor([list(graph.edge_index[1]), list(graph.edge_index[0])])
    l1 = top_sort(ei2, graph.num_nodes)
    ns = torch.LongTensor([i for i in range(graph.num_nodes)])

    graph.__setattr__("bi_layer_idx0", l0)
    graph.__setattr__("bi_layer_index0", ns)
    graph.__setattr__("bi_layer_idx1", l1)
    graph.__setattr__("bi_layer_index1", ns)

    assert_order(graph.edge_index, l0, ns)
    assert_order(ei2, l1, ns)

import networkx as nx
from torch import Tensor
import numpy as np
from numpy import linalg as la
import dgl
import matplotlib.pyplot as plt
from pandas import DataFrame
from IPython.display import display

from src.arch import DAGConv, FB_DAGConv, SF_DAGConv
from src.baselines_archs import GCNN_2L, GCNN, GAT, MLP, MyGCNN
import src.dag_utils as dagu


def get_graph_data(d_dat_p):
    Adj, dag = dagu.create_dag(d_dat_p['N'], d_dat_p['p'])
    W = la.inv(np.eye(d_dat_p['N']) - Adj)
    W_inf = la.inv(W)
    GSOs = np.array([(W * dagu.compute_Dq(dag, i, d_dat_p['N'])) @ W_inf for i in range(d_dat_p['N'])])
    return Adj, W, GSOs

def kipf_GSO(S):
    S_hat = S + np.eye(S.shape[0])
    Deg_sqrt_inv = np.diag(1/np.sqrt(S_hat.sum(axis=0)))
    return Deg_sqrt_inv @ S_hat @ Deg_sqrt_inv


def select_GSO(arc_p, GSOs, sel_GSOs, W, Adj):
    if arc_p['GSO'] == 'GSOs':
        return Tensor(GSOs)
    elif arc_p['GSO'] == 'sel_GSOs':
        return Tensor(sel_GSOs)
    elif arc_p['GSO'] == 'rnd_GSOs':
        rnd_idx = np.random.choice(GSOs.shape[1], size=arc_p['n_gsos'], replace=False)
        return  Tensor(GSOs[rnd_idx])
    elif arc_p['GSO'] == 'first_GSOs':
        return Tensor(GSOs[:arc_p['n_gsos']])
    elif arc_p['GSO'] == 'last_GSOs':
        return Tensor(GSOs[-arc_p['n_gsos']:])
    elif arc_p['GSO'] == 'W-dgl':
        return dgl.from_networkx(nx.from_numpy_array(W)).add_self_loop()
    elif arc_p['GSO'] == 'A-dgl':
        return dgl.from_networkx(nx.from_numpy_array(Adj)).add_self_loop()
    elif arc_p['GSO'] == 'W':
        return Tensor(kipf_GSO(W))
    elif arc_p['GSO'] == 'A':
        return Tensor(kipf_GSO(Adj))
    else:
        return None
    

def instantiate_arch(arc_p, K):
    if arc_p['arch'] in [DAGConv, FB_DAGConv]:
        return arc_p['arch'](arc_p['in_dim'], arc_p['hid_dim'], arc_p['out_dim'], K, arc_p['L'],
                             last_act=arc_p['l_act'])

    elif arc_p['arch'] == SF_DAGConv:
        return arc_p['arch'](arc_p['in_dim'], arc_p['out_dim'], K, arc_p['L'], last_act=arc_p['l_act'])

    elif arc_p['arch'] in [GCNN_2L, MyGCNN]:
        return arc_p['arch'](arc_p['in_dim'], arc_p['hid_dim'], arc_p['out_dim'],
                             last_act=arc_p['l_act'])
    
    elif arc_p['arch'] == GCNN:
        return arc_p['arch'](arc_p['in_dim'], arc_p['hid_dim'], arc_p['out_dim'], arc_p['L'],
                             last_act=arc_p['l_act'])
    
    elif arc_p['arch'] == GAT:
        return arc_p['arch'](arc_p['in_dim'], arc_p['hid_dim'], arc_p['out_dim'], arc_p['n_heads'],
                             arc_p['gat_params'], last_act=arc_p['l_act'])

    elif arc_p['arch'] == MLP:
        return arc_p['arch'](arc_p['in_dim'], arc_p['hid_dim'], arc_p['out_dim'], last_act=arc_p['l_act'])

    else:
        raise Exception('Unknown architecture type')
    

def display_data(exps_leg, err, std, time, metric_label='Err'):
    data = {
        'Exp': exps_leg,
        f'Mean {metric_label}': err.mean(axis=0),
        f'Median {metric_label}': np.median(err, axis=0),
        'Mean Std': std.mean(axis=0) if std is not None else None,
        'time': time.mean(axis=0)
    }
    df = DataFrame(data)
    display(df)


def plot_results(err, x_values, exps, xlabel, ylabel='Mean Err', figsize=(8,5), skip_idx=[],
                 logy=True, n_cols=3, ylim_bottom=1e-2, ylim_top=1):
    plt.figure(figsize=figsize)

    for i, exp in enumerate(exps):
        if i in skip_idx:
            continue
        if logy:
            plt.semilogy(x_values, err[:,i], exp['fmt'], label=exp['leg'], linewidth=2.0)
        else:
            plt.plot(x_values, err[:,i], exp['fmt'], label=exp['leg'], linewidth=2.0)
    
    plt.ylim(ylim_bottom, ylim_top)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(ncol=n_cols)
    plt.grid(True)
    plt.show()

def load_data(file_name):
    data = np.load(file_name, allow_pickle=True)
    err = data['err']
    std = data['std']
    times = data['times']
    Exps = data['exp']
    xvals = data['xvals']
    return err, std, times, Exps, xvals
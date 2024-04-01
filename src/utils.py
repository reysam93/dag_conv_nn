import networkx as nx
from torch import Tensor
import numpy as np
import dgl
import matplotlib.pyplot as plt
from pandas import DataFrame
from IPython.display import display

from src.arch import DAGConv, FB_DAGConv, SF_DAGConv
from src.baselines_archs import GCNN_2L, GCNN, GAT, MLP

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
    elif arc_p['GSO'] == 'W':
        return dgl.from_networkx(nx.from_numpy_array(W)).add_self_loop()  #.to(device)
    elif arc_p['GSO'] == 'A':
        return dgl.from_networkx(nx.from_numpy_array(Adj)).add_self_loop()  #.to(device)
    else:
        return None
    

def instantiate_arch(arc_p, K):
    if arc_p['arch'] in [DAGConv, FB_DAGConv]:
        return arc_p['arch'](arc_p['in_dim'], arc_p['hid_dim'], arc_p['out_dim'], K, arc_p['L'])

    elif arc_p['arch'] == SF_DAGConv:
        return arc_p['arch'](arc_p['in_dim'], arc_p['out_dim'], K, arc_p['L'])

    elif arc_p['arch'] == GCNN_2L:
        return arc_p['arch'](arc_p['in_dim'], arc_p['hid_dim'], arc_p['out_dim'])
    
    elif arc_p['arch'] == GCNN:
        return arc_p['arch'](arc_p['in_dim'], arc_p['hid_dim'], arc_p['out_dim'], arc_p['L'])
    
    elif arc_p['arch'] == GAT:
        return arc_p['arch'](arc_p['in_dim'], arc_p['hid_dim'], arc_p['out_dim'], arc_p['n_heads'],
                             arc_p['gat_params'])

    elif arc_p['arch'] == MLP:
        return arc_p['arch'](arc_p['in_dim'], arc_p['hid_dim'], arc_p['out_dim'])

    else:
        raise Exception('Unknown architecture type')
    

def display_data(exps_leg, err, std, time):
    data = {
        'Exp': exps_leg,
        'Mean Err': err.mean(axis=0),
        'Median Err': np.median(err, axis=0),
        'Mean Std': std.mean(axis=0),
        'time': time.mean(axis=0)
    }
    df = DataFrame(data)
    display(df)


def plot_results(err, x_values, exps, xlabel, ylabel='Mean Err', figsize=(10,6), skip_idx=[],
                 logy=True, n_cols=3):
    plt.figure(figsize=figsize)

    for i, exp in enumerate(exps):
        if i in skip_idx:
            continue
        if logy:
            plt.semilogy(x_values, err[:,i], exp['fmt'], label=exp['leg'], linewidth=2.0)
        else:
            plt.plot(x_values, err[:,i], exp['fmt'], label=exp['leg'], linewidth=2.0)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(ncol=n_cols)
    plt.grid(True)
    plt.show()
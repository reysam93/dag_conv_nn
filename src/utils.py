import networkx as nx
from torch import Tensor
import numpy as np
from numpy import linalg as la
import dgl
import matplotlib.pyplot as plt
from pandas import DataFrame
from IPython.display import display

from src.arch import DAGConv, FB_DAGConv, ADCN
from src.baselines_archs import GAT, MLP, MyGCNN, GraphSAGE, GIN
import src.dag_utils as dagu


def get_graph_data(d_dat_p, get_Psi=False):
    Adj, dag = dagu.create_dag(d_dat_p['N'], d_dat_p['p'])
    W = la.inv(np.eye(d_dat_p['N']) - Adj)
    W_inf = la.inv(W)

    if get_Psi:
        Psi = np.array([dagu.compute_Dq(dag, i, d_dat_p['N']) for i in range(d_dat_p['N'])]).T
        GSOs = np.array([(W * Psi[:,i]) @ W_inf for i in range(d_dat_p['N'])])
        return Adj, W, GSOs, Psi
    
    GSOs = np.array([(W * dagu.compute_Dq(dag, i, d_dat_p['N'])) @ W_inf for i in range(d_dat_p['N'])])
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
    else:
        return None
    

def instantiate_arch(arc_p, K):
    args = arc_p['args']
    if arc_p['arch'] in [DAGConv, FB_DAGConv]:
        args['K'] = K

    elif arc_p['arch'] == GAT:
        args = {k: v for k, v in args.items() if k not in ['bias', 'n_layers']}

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
import numpy as np
import time
# import dgl
import torch
from torch import nn
import networkx as nx
from tqdm.auto import tqdm
import sys
import os
import pickle
from numpy import linalg as la
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import random
from torch_geometric.data import Data


import src.dag_utils as dagu
import src.utils as utils
from src.arch import SMLP, DAGConv, FB_DAGConv, ParallelMLPSum, SAM1
from src.models import Model, LinDAGRegModel
# from src.baselines_archs import GAT, MLP, MyGCNN, GraphSAGE, GIN
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="dgl")
import os
import igraph as ig
# from src.utils_Dagnn import *
# from src.DAGNN import DAGNN, DVAE



# Ser random seed
SEED = 10
PATH = 'results/diffusion/'
SAVE = True
np.random.seed(SEED)
torch.manual_seed(SEED)
# dgl.random.seed(SEED)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
# torch.cuda.set_per_process_memory_fraction(.5, device=device)



M = 400

data_p = {
    'n_tries': 1,  #25,

    ## Graph parameters
    'p': 0.2,  # .2                  # Edge prob in Erdos-Renyi DAG
    'N': 25,                    # Number of nodes

    ## Signal parameters
    'M': M,                   # Number of observed signals
    'M_train': int(0.7 * M),  # Samples selected for training
    'M_val': int(0.2 * M),    # Samples selected for validation
    'M_test': int(0.1 * M),   # Samples selected for test
    'src_t': 'constant',          # 'random' or 'constant'
    'max_src_node': 15, #25,           # Maximum index of nodes allowed to be sources
    'n_sources': 14,             # Maximum Number of source nodes
    'n_p_x': .1,
    'n_p_y': .1,                 # Normalized noise power
    'max_GSO': 25,              # Maximum index of GSOs involved in the diffusion
    'min_GSO': 5,               # Minimum index of GSOs involved in the diffusion
    'n_GSOs': 10,            # Number of GSOs
}

default_mod_p = {
    'bs': 100,           # Size of the batch
    'lr': 5e-3,         # Learning rate
    'epochs': 200,  #50,       # Number of training epochs 
    'pat': 25,  # 15        # Number of non-decreasing epoch to stop training
    'wd': 1e-4,         # Weight decay
}

default_arch_args = {
    'in_dim': 1,        # Input dimension
    'hid_dim': 32,     # Hidden dimension
    'out_dim': 1,       # Output dimension
    'n_layers': 2,#2,  # 3 also works well          # Number of layers
    'l_act': None,
    'bias': True,
}


def add_noise(signal, n_p):
    shape = signal.shape

    M = shape[0]
    N = shape[1]
 
    if n_p <= 0:
        return signal

    signal_norm = torch.norm(signal, p=2, dim=1, keepdim=True)
    signal_norm[signal_norm == 0] = 1
    noise = torch.randn(M, N, 1, device=signal.device)
    noise_norm = torch.norm(noise, p=2, dim=1, keepdim=True)
    noise = noise * signal_norm * torch.sqrt(torch.tensor(n_p)) / noise_norm
    
    return signal + noise



def get_real_data(d_dat_p ,get_Psi=False):


    Adj = np.load('adj_matrix.npy')  


    def is_triangular(A):
        if not isinstance(A, torch.Tensor):
            A = torch.tensor(A)
            
        # Get upper and lower triangular parts
        upper_tri = A.triu(1)
        lower_tri = A.tril(-1)
        
        # Check if equal to either upper or lower triangular part
        is_upper = torch.equal(A, upper_tri)
        is_lower = torch.equal(A, lower_tri)
        
        return is_upper, is_lower

    is_upper, is_lower = is_triangular(Adj)
    print("Is upper triangular:", is_upper)
    print("Is lower triangular:", is_lower)


    W = la.inv(np.eye(d_dat_p['N']) - Adj)
    W_inf = la.inv(W)
    dag = nx.from_numpy_array(Adj.T, create_using=nx.DiGraph())
    if get_Psi:
        Psi = np.array([dagu.compute_Dq(dag, i, d_dat_p['N']) for i in range(d_dat_p['N'])]).T
        GSOs = np.array([(W * Psi[:,i]) @ W_inf for i in range(d_dat_p['N'])])
        return Adj, W, GSOs, Psi
    
    GSOs = np.array([(W * dagu.compute_Dq(dag, i, d_dat_p['N'])) @ W_inf for i in range(d_dat_p['N'])])

    return Adj, W, GSOs



def arth(d_p, GSOs):

    range_GSO = np.arange(d_p['min_GSO'], d_p['max_GSO'])
    gsos_idx = np.random.choice(range_GSO, size=d_p['n_GSOs'], replace=False)
    sel_GSOs = GSOs[gsos_idx]
    Y = torch.from_numpy(np.load('X2_8_12.npy')).float()

    X = Y.clone()
    X = X.unsqueeze(2)
    Y = Y.unsqueeze(2)

    X[:, -10, :] = 0
    Xn_t = add_noise(X, 0.0)
    Yn_t = add_noise(Y, 0.0)
    X_data = {'train': Xn_t[:d_p['M_train']], 'val': Xn_t[d_p['M_train']:-d_p['M_test']], 'test': Xn_t[-d_p['M_test']:]}
    Y_data = {'train': Yn_t[:d_p['M_train']], 'val': Yn_t[d_p['M_train']:-d_p['M_test']], 'test': Y[-d_p['M_test']:]}

    return X_data, Y_data, sel_GSOs, gsos_idx
  

def get_signals(d_p, GSOs):
    range_GSO = np.arange(d_p['min_GSO'], d_p['max_GSO'])
    gsos_idx = np.random.choice(range_GSO, size=d_p['n_GSOs'], replace=False)
    sel_GSOs = GSOs[gsos_idx]
    Yn_t, X_t, Y_t = dagu.create_diff_data(d_p['M'], sel_GSOs, d_p['max_src_node'], d_p['n_p_x'], d_p['n_p_y'],
                                           d_p['n_sources'], src_t=d_p['src_t'], torch_tensor=True, verb=False)
    

    
    X_data = {'train': X_t[:d_p['M_train']], 'val': X_t[d_p['M_train']:-d_p['M_test']], 'test': X_t[-d_p['M_test']:]}
    Y_data = {'train': Yn_t[:d_p['M_train']], 'val': Yn_t[d_p['M_train']:-d_p['M_test']],
              'test': Y_t[-d_p['M_test']:]}
        
    return X_data, Y_data, sel_GSOs, gsos_idx


def run_exp(d_p, d_arc_args, d_mod_p, exps, verb=True):
    # Create error variables
    err = np.zeros((d_p['n_tries'], len(exps)))
    std = np.zeros((d_p['n_tries'], len(exps)))
    times = np.zeros((d_p['n_tries'], len(exps)))

    t_begin = time.time()
    # for i in range(d_p['n_tries']):
    with tqdm(total=d_p['n_tries']*len(exps), disable=False) as pbar:
        for i in range(d_p['n_tries']):
            Adj, W, GSOs, Psi = get_real_data(d_p, get_Psi=True)

            X_data, Y_data, sel_GSOs, sel_GSOs_idx = arth(d_p, GSOs)

            for j, exp in enumerate(exps):
                arc_p = {**exp['arc_p']}
                
                arc_p['args'] = {**d_arc_args, **arc_p['args']} if 'args' in arc_p.keys() else {**d_arc_args}
                mod_p = {**d_mod_p, **exp['mod_p']} if 'mod_p' in exp.keys() else d_mod_p

                GSO = utils.select_GSO(arc_p, GSOs, sel_GSOs, W, Adj)
                K = GSO.shape[0] if isinstance(GSO, torch.Tensor) and len(GSO.shape) == 3 else 0  
                arch = utils.instantiate_arch(arc_p, K)  
                model = Model(arch, device=device)
                t_i = time.time()
                model.fit(X_data, Y_data, GSO, mod_p['lr'], mod_p['epochs'], mod_p['bs'], mod_p['wd'], patience=mod_p['pat'])
                t_e = time.time() - t_i
                err[i,j], std[i,j] = model.test(X_data['test'], Y_data['test'], GSO)
                params = arch.n_params if hasattr(arch, 'n_params') else None 

                times[i,j] = t_e
                
                # params = arch.n_params if hasattr(arch, 'n_params') else None 

                # Progress
                pbar.update(1)
                if verb:
                    print(f'-{i}. {exp["leg"]}: err: {err[i,j]:.3f} | std: {std[i,j]:.3f}  |' +
                          f' time: {times[i,j]:.1f} | n_params: {params}')
                    
    total_t = (time.time() - t_begin)/60
    print(f'----- Ellapsed time: {total_t:.2f} minutes -----')
    return err, std, times



Exps = [
    # Our Models 

    {'arc_p': {'arch': ParallelMLPSum , 'GSO': 'GSOs', 'n_inputs': 25, 'input_dim': 25, 'hidden_dims': [32], 'output_dim': 25}, 'leg': 'ParallelMLPSum - 1 layer, hid_dim:32'},
    # {'arc_p': {'arch': SharedMLPSum , 'GSO': 'GSOs', 'n_inputs': 20, 'input_dim': 20, 'hidden_dims': [32], 'output_dim': 20}, 'leg': 'SharedMLPSum - 1 layer, hid_dim:32'},
    # {'arc_p': {'arch': SMLP , 'GSO': 'GSOs', 'in_dim': 1, 'hid_dim': [64], 'out_dim': 1, 'bias' : True }, 'leg': 'SMLP2-32'},



    {'arc_p': {'arch': SMLP , 'GSO': 'GSOs', 'in_dim': 1, 'hid_dim': [32], 'out_dim': 1, 'bias' : True }, 'leg': 'SMLP-32'},
    # {'arc_p': {'arch': SMLP , 'GSO': 'GSOs', 'in_dim': 1, 'hid_dim': [512], 'out_dim': 1, 'bias' : True }, 'leg': 'SMLP-512'},
    # {'arc_p': {'arch': SAM1 , 'GSO': 'GSOs', 'n_inputs': 10, 'input_dim': 1, 'hidden_dims': [128], 'output_dim': 1}, 'leg': 'SAM1'},
    # {'arc_p': {'arch': SMLP , 'GSO': 'GSOs', 'in_dim': 1, 'hid_dim': [64], 'out_dim': 1, 'bias' : True }, 'leg': 'SMLP-16'},

    ]



mod_p_init = default_mod_p.copy()
mod_p_init['pat'] = 50
verb = True
err, std, times = run_exp(data_p, default_arch_args, mod_p_init, Exps, verb=verb)


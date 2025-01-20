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
from src.arch import DAGConv, FB_DAGConv, SF_DAGConv, ADCN , ParallelMLPSum, SharedMLPSum, SMLP
from src.models import Model, LinDAGRegModel, Model3
from src.baselines_archs import GAT, MLP, MyGCNN, GraphSAGE, GIN
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="dgl")
import os
import igraph as ig
from src.utils_Dagnn import *
from src.DAGNN import DAGNN, DVAE


  
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



M = 343

data_p = {
    'n_tries': 4,  #25,

    ## Graph parameters
    'p': 0.2,  # .2                  # Edge prob in Erdos-Renyi DAG
    'N': 20,                    # Number of nodes

    ## Signal parameters
    'M': M,                   # Number of observed signals
    'M_train': int(0.7 * M),  # Samples selected for training
    'M_val': int(0.2 * M),    # Samples selected for validation
    'M_test': int(0.1 * M),   # Samples selected for test
    'src_t': 'constant',          # 'random' or 'constant'
    'max_src_node': 15, #25,           # Maximum index of nodes allowed to be sources
    'n_sources': 5,             # Maximum Number of source nodes
    'n_p_x': .05,
    'n_p_y': .05,                 # Normalized noise power
    'max_GSO': 20,              # Maximum index of GSOs involved in the diffusion
    'min_GSO': 5,               # Minimum index of GSOs involved in the diffusion
    'n_GSOs': 10,            # Number of GSOs
    'concentration': 'silicon.npy'
}

default_mod_p = {
    'bs': 25,           # Size of the batch
    'lr': 5e-4,         # Learning rate
    'epochs': 50,  #50,       # Number of training epochs 
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
    nodes = ['CH','CL','CN','CU','EN','EV','LE','LO','OC','PA','RA','TH','TM','WI','KE','TN','TS','TW','TSO','TR']
    G1 = nx.DiGraph()
    G1.add_nodes_from(nodes)
    edges = [
        ("TH", "TN"), ("CL", "TN"), ("LE", "TN"), ("CN", "TN"), ("WI", "TN"),
        ("EV", "TS"), ("TM", "TW"), ("OC", "TW"), ("CH", "TW"), ("RA", "TW"),
        ("EN", "KE"), ("TN", "TS"), ("TS", "TW"), ("TW", "TSO"),("KE", "TSO"),
        ("PA", "TSO"),("TSO", "TR"), ("LO", "TR"), ("CU", "TR")
    ]
    G1.add_edges_from(edges)
    assert nx.is_directed_acyclic_graph(G1), "The graph must be a DAG."
    topo_order = list(nx.topological_sort(G1))
    Adj = nx.to_numpy_array(G1, nodelist=topo_order, dtype=int)
    W = la.inv(np.eye(d_dat_p['N']) - Adj.T)
    W_inf = la.inv(W)
    dag = nx.from_numpy_array(Adj, create_using=nx.DiGraph())
    if get_Psi:
        Psi = np.array([dagu.compute_Dq(dag, i, d_dat_p['N']) for i in range(d_dat_p['N'])]).T
        GSOs = np.array([(W * Psi[:,i]) @ W_inf for i in range(d_dat_p['N'])])
        return Adj, W, GSOs, Psi
    GSOs = np.array([(W * dagu.compute_Dq(dag, i, d_dat_p['N'])) @ W_inf for i in range(d_dat_p['N'])])

    return Adj, W, GSOs


def is_upper_triangular(adj_matrix):
    return np.allclose(adj_matrix, np.triu(adj_matrix))

def randomize_dag_adj(A_true, alpha):
    num_nodes = A_true.shape[0]
    
    while True:
        A_rand = np.triu(A_true.copy())  
    
        total_possible_edges = (num_nodes * (num_nodes - 1)) // 2
        num_modifications = int(alpha * total_possible_edges)
        
        possible_edges = [(i, j) for i in range(num_nodes) for j in range(i+1, num_nodes)]
        
        modifications = 0
        while modifications < num_modifications:
            # Randomly choose to add or remove an edge
            action = random.choice(['add', 'remove'])
            i, j = random.choice(possible_edges)
            
            if action == 'add' and A_rand[i, j] == 0:
                A_rand[i, j] = 1
                modifications += 1
            elif action == 'remove' and A_rand[i, j] == 1:
                A_rand[i, j] = 0
                modifications += 1
        
        if is_upper_triangular(A_rand):
            return A_rand


def Thames(d_p,GSOs, pollutant=None):

    range_GSO = np.arange(d_p['min_GSO'], d_p['max_GSO'])
    gsos_idx = np.random.choice(range_GSO, size=d_p['n_GSOs'], replace=False)
    sel_GSOs = GSOs[gsos_idx]
    t1 = np.load(pollutant)
    Yn_t = torch.tensor(t1, dtype = torch.float32)

    Xn_t = Yn_t.clone()
    Xn_t[:,-6:,:]  = 0
    
    Xn_t = add_noise(Xn_t, 0.0005)

    X_data = {'train': Xn_t[:d_p['M_train']], 'val': Xn_t[d_p['M_train']:-d_p['M_test']], 'test': Xn_t[-d_p['M_test']:]}
    Y_data = {'train': Yn_t[:d_p['M_train']], 'val': Yn_t[d_p['M_train']:-d_p['M_test']], 'test': Yn_t[-d_p['M_test']:]}

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
    print(d_p['concentration'])
    err = np.zeros((d_p['n_tries'], len(exps)))
    std = np.zeros((d_p['n_tries'], len(exps)))
    times = np.zeros((d_p['n_tries'], len(exps)))

    t_begin = time.time()
    # for i in range(d_p['n_tries']):
    with tqdm(total=d_p['n_tries']*len(exps), disable=False) as pbar:
        for i in range(d_p['n_tries']):
            Adj, W, GSOs, Psi = get_real_data(d_p, get_Psi=True)

            X_data, Y_data, sel_GSOs, sel_GSOs_idx = Thames(d_p, GSOs, d_p['concentration'])

            for j, exp in enumerate(exps):
                arc_p = {**exp['arc_p']}
                
                arc_p['args'] = {**d_arc_args, **arc_p['args']} if 'args' in arc_p.keys() else {**d_arc_args}
                mod_p = {**d_mod_p, **exp['mod_p']} if 'mod_p' in exp.keys() else d_mod_p

                if exp['arc_p']['arch'] == LinDAGRegModel:
                    # Fit and test linear model
                    if 'transp' in arc_p.keys() and arc_p['transp']:
                        dag_T = nx.from_numpy_array(Adj, create_using=nx.DiGraph())
                        Psi = np.array([dagu.compute_Dq(dag_T, i, d_p['N']) for i in range(d_p['N'])]).T
                        arc_p['transp'] = False

                    Psi_sel = utils.select_GSO(arc_p, Psi.T, Psi[:,sel_GSOs_idx].T, W, Adj, sel_GSOs_idx).numpy().T
                    lin_model = LinDAGRegModel(W, Psi_sel)
                    t_i = time.time()
                    lin_model.fit(X_data['train'], Y_data['train'])
                    t_e = time.time() - t_i
                    err[i,j], std[i,j] = lin_model.test(X_data['test'], Y_data['test'])

                elif exp['arc_p']['arch'] == DVAE:
                    X_data1 = DVAE_exp(Adj, X_data)
                    GSO = utils.select_GSO(arc_p, GSOs, sel_GSOs, W, Adj)
                    K = GSO.shape[0] if isinstance(GSO, torch.Tensor) and len(GSO.shape) == 3 else 0  
                    arch = utils.instantiate_arch(arc_p, K)      
                    model = Model(arch, device=device)
                    t_i = time.time()
                    model.fit(X_data1, Y_data, GSO, mod_p['lr'], mod_p['epochs'], mod_p['bs'], mod_p['wd'], patience=mod_p['pat'])
                    t_e = time.time() - t_i
                    err[i,j], std[i,j] = model.test(X_data1['test'], Y_data['test'], GSO)

                elif exp['arc_p']['arch'] == DAGNN:
                    X_data1, Y_data1 = DAGNN_prep(Adj, X_data, Y_data)
                    GSO = utils.select_GSO(arc_p, GSOs, sel_GSOs, W, Adj)
                    K = GSO.shape[0] if isinstance(GSO, torch.Tensor) and len(GSO.shape) == 3 else 0  
                    arch = utils.instantiate_arch(arc_p, K)      
                    model = Model(arch, device=device)
                    t_i = time.time()
                    model.fit(X_data1, Y_data1, GSO, mod_p['lr'], mod_p['epochs'], mod_p['bs'], mod_p['wd'], patience=mod_p['pat'])
                    t_e = time.time() - t_i
                    err[i,j], std[i,j] = model.test(X_data1['test'], Y_data1['test'], GSO)
                else:
                    # Fit and test nonlinear models
                    GSO = utils.select_GSO(arc_p, GSOs, sel_GSOs, W, Adj)
                    K = GSO.shape[0] if isinstance(GSO, torch.Tensor) and len(GSO.shape) == 3 else 0  
                    arch = utils.instantiate_arch(arc_p, K)  
                    model = Model(arch, device=device)
                    t_i = time.time()
                    model.fit(X_data, Y_data, GSO, mod_p['lr'], mod_p['epochs'], mod_p['bs'], mod_p['wd'], patience=mod_p['pat'])
                    t_e = time.time() - t_i
                    err[i,j], std[i,j] = model.test(X_data['test'], Y_data['test'], GSO)
                times[i,j] = t_e
                
                params = arch.n_params if hasattr(arch, 'n_params') else None 

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


    # {'arc_p': {'arch': ParallelMLPSum , 'GSO': 'GSOs', 'n_inputs': 20, 'input_dim': 20, 'hidden_dims': [32], 'output_dim': 20}, 'leg': 'ParallelMLPSum - 1 layer, hid_dim:32'},
    # {'arc_p': {'arch': SharedMLPSum , 'GSO': 'GSOs', 'n_inputs': 20, 'input_dim': 20, 'hidden_dims': [32], 'output_dim': 20}, 'leg': 'SharedMLPSum - 1 layer, hid_dim:32'},
    {'arc_p': {'arch': SMLP , 'GSO': 'GSOs', 'in_dim': 1, 'hid_dim': [128], 'out_dim': 1, 'bias' : True }, 'leg': 'SMLP-128'},
    {'arc_p': {'arch': SMLP , 'GSO': 'GSOs', 'in_dim': 1, 'hid_dim': [256], 'out_dim': 1, 'bias' : True }, 'leg': 'SMLP-256'},
    {'arc_p': {'arch': SMLP , 'GSO': 'GSOs', 'in_dim': 1, 'hid_dim': [512], 'out_dim': 1, 'bias' : True }, 'leg': 'SMLP-512'},


    # {'arc_p': {'arch': ParallelMLPSum , 'GSO': 'GSOs', 'n_inputs': 20, 'input_dim': 20, 'hidden_dims': [64], 'output_dim': 20}, 'leg': 'ParallelMLPSum - 1 layer, hid_dim:64'},
    # {'arc_p': {'arch': SharedMLPSum , 'GSO': 'GSOs', 'n_inputs': 20, 'input_dim': 20, 'hidden_dims': [64], 'output_dim': 20}, 'leg': 'SharedMLPSum - 1 layer, hid_dim:64'},


    {'arc_p': {'arch': FB_DAGConv, 'GSO': 'GSOs'}, 'leg': 'DCN'},
    # {'arc_p': {'arch': DAGNN, 'GSO': 'GSOs', 'emb_dim': 1, 'hidden_dim':501, 'out_dim': 501, 'max_n': 20,'nvt':1 ,
    #            'START_TYPE': 0, 'END_TYPE': 1, 'hs':501, 'nz': 56, 'agg': "attn_h", 'num_layers':2, 'bidirectional': False, 'out_wx': False, 'out_pool_all': False, 
    #            'out_pool': P_MAX, 'dropout': 0, 'num_nodes': 20}, 'leg': 'DAGNN'},
    # {'arc_p': {'arch': DVAE, 'GSO': 'GSOs','max_n':20, 'nvt':1, 'START_TYPE':0, 'END_TYPE':1, 'hs':501,'nz':20, 'bidirectional': True, 'vid': True}, 'leg': 'DVAE'},

    # {'arc_p': {'arch': FB_DAGConv, 'GSO': 'rnd_GSOs', 'n_gsos': 15}, 'leg': 'DCN-15'},
    # {'arc_p': {'arch': FB_DAGConv, 'GSO': 'rnd_GSOs', 'n_gsos': 10}, 'leg': 'DCN-10'},
    # {'arc_p': {'arch': FB_DAGConv, 'GSO': 'rnd_GSOs', 'n_gsos': 5}, 'leg': 'DCN-5'},

    # {'arc_p': {'arch': DAGConv, 'GSO': 'GSOs'}, 'leg': 'DAGConv'},
    # {'arc_p': {'arch': DAGConv, 'GSO': 'rnd_GSOs', 'n_gsos': 15}, 'leg': 'DAGConv-15'},
    # {'arc_p': {'arch': DAGConv, 'GSO': 'rnd_GSOs', 'n_gsos': 10}, 'leg': 'DAGConv-10'},
    # {'arc_p': {'arch': DAGConv, 'GSO': 'rnd_GSOs', 'n_gsos': 5}, 'leg': 'DAGConv-5'},


    # {'arc_p': {'arch': FB_DAGConv, 'GSO': 'GSOs', 'transp': True}, 'leg': 'DCN-T'},
    # {'arc_p': {'arch': FB_DAGConv, 'GSO': 'rnd_GSOs', 'n_gsos': 15, 'transp': True}, 'leg': 'DCN-15-T'},
    # {'arc_p': {'arch': FB_DAGConv, 'GSO': 'rnd_GSOs', 'n_gsos': 10, 'transp': True}, 'leg': 'DCN-10-T'},

    # {'arc_p': {'arch': DAGConv, 'GSO': 'GSOs', 'transp': True}, 'leg': 'DAGConv-T'},
    # {'arc_p': {'arch': DAGConv, 'GSO': 'rnd_GSOs', 'n_gsos': 15, 'transp': True}, 'leg': 'DAGConv-15-T'},

    # {'arc_p': {'arch': ADCN, 'GSO': 'GSOs', 'transp': True, 'args': {'mlp_layers': 4}}, 'leg': 'ADCN-4-T'},
    # {'arc_p': {'arch': ADCN, 'GSO': 'GSOs', 'transp': True, 'args': {'mlp_layers': 5}}, 'leg': 'ADCN-5-T'},


    # {'arc_p': {'arch': LinDAGRegModel, 'GSO': 'GSOs'}, 'leg': 'Linear'},
    # {'arc_p': {'arch': LinDAGRegModel, 'GSO': 'rnd_GSOs', 'n_gsos': 15}, 'leg': 'Linear-15'},
    # {'arc_p': {'arch': LinDAGRegModel, 'GSO': 'rnd_GSOs', 'n_gsos': 10}, 'leg': 'Linear-10'},
    # {'arc_p': {'arch': LinDAGRegModel, 'GSO': 'GSOs', 'transp': True}, 'leg': 'Linear-T'},
    # {'arc_p': {'arch': LinDAGRegModel, 'GSO': 'rnd_GSOs', 'n_gsos': 15, 'transp': True}, 'leg': 'Linear-15-T'},
    # {'arc_p': {'arch': LinDAGRegModel, 'GSO': 'rnd_GSOs', 'n_gsos': 10, 'transp': True}, 'leg': 'Linear-10-T'},


    # {'arc_p': {'arch': ADCN, 'GSO': 'GSOs', 'args': {'mlp_layers': 4}}, 'leg': 'ADCN-4-2'},
    # {'arc_p': {'arch': ADCN, 'GSO': 'GSOs', 'args': {'mlp_layers': 5}}, 'leg': 'ADCN-5-2'},
    # {'arc_p': {'arch': ADCN, 'GSO': 'GSOs', 'args': {'mlp_layers': 4, 'n_layers': 4}}, 'leg': 'ADCN-4-4'},
    # {'arc_p': {'arch': ADCN, 'GSO': 'GSOs', 'args': {'mlp_layers': 5, 'n_layers': 4}}, 'leg': 'ADCN-5-4'},
    

    # {'arc_p': {'arch': GraphSAGE, 'GSO': 'A-dgl', 'args': {'aggregator': 'mean'}}, 'leg': 'GraphSAGE-A'},
    # {'arc_p': {'arch': GIN, 'GSO': 'A-dgl', 'args': {'aggregator': 'sum'}}, 'leg': 'GIN-A'},
    # {'arc_p': {'arch': GIN, 'GSO': 'A-dgl', 'args': {'aggregator': 'sum', 'mlp_layers': 4}}, 'leg': 'GIN-A-4'},
    # {'arc_p': {'arch': MLP, 'GSO': None}, 'leg': 'MLP'},
    # {'arc_p': {'arch': MLP, 'GSO': None, 'args': {'n_layers': 4}}, 'leg': 'MLP-4'},
    # {'arc_p': {'arch': MyGCNN, 'GSO': 'A'}, 'leg': 'GNN-A'},

    # {'arc_p': {'arch': FB_DAGConv, 'GSO': 'A_pows', 'K': 2, 'transp': False}, 'leg': 'FB-GCNN-2'},
    # {'arc_p': {'arch': FB_DAGConv, 'GSO': 'A_pows', 'K': 3, 'transp': False}, 'leg': 'FB-GCNN-3'},
    # {'arc_p': {'arch': FB_DAGConv, 'GSO': 'A_pows', 'K': 4, 'transp': False}, 'leg': 'FB-GCNN-4'},
    # {'arc_p': {'arch': GAT, 'GSO': 'A-dgl', 'args': {'num_heads': 2, 'hid_dim': 16, 'gat_params': {'attn_drop': 0}}},
    #  'leg': 'GAT'},

    ]

mod_p_init = default_mod_p.copy()
mod_p_init['pat'] = 50
verb = True
err, std, times = run_exp(data_p, default_arch_args, mod_p_init, Exps, verb=verb)

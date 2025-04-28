import numpy as np
import time
import dgl
import torch
from torch.nn import functional as F
from torch import nn
import networkx as nx
from tqdm.auto import tqdm

import src.dag_utils as dagu
import src.utils as utils
from src.arch import DAGConv, FB_DAGConv, SF_DAGConv, ADCN, SMLP
from src.models import SrcIdModel, LinDAGClassModel
from src.baselines_archs import MyGCNN, GAT, MLP, GraphSAGE, GIN
from src.DAGNN import DAGNN, DVAE
from src.utils_Dagnn import *

# Ser random seed
SEED = 10
PATH = 'results/src_id/'
SAVE = True
np.random.seed(SEED)
torch.manual_seed(SEED)
dgl.random.seed(SEED)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



M = 2000

data_p = {
    'n_tries': 1,  #25,

    ## Graph parameters
    'p': 0.2,                    # Edge prob in Erdos-Renyi DAG
    'N': 100,                    # Number of nodes

    ## Signal parameters
    'M': M,                   # Number of observed signals
    'M_train': int(0.7 * M),  # Samples selected for training
    'M_val': int(0.2 * M),    # Samples selected for validation
    'M_test': int(0.1 * M),   # Samples selected for test
    'src_t': 'random',          # 'random' or 'constant'
    'max_src_node': 3,  # 10           # Maximum index of nodes allowed to be sources
    'n_sources': 1,              # Maximum Number of source nodes
    'n_p_y': .05,  
    'n_p_x': 0,                  # Normalized noise power
    'max_GSO': 100,              # Maximum index of GSOs involved in the diffusion
    'min_GSO': 50,               # Minimum index of GSOs involved in the diffusion
    'n_GSOs': 50                # Number of GSOs
}

# Model parameters
default_arch_args = {
    'in_dim': 1,        # Input dimension
    'hid_dim': 32,     # Hidden dimension
    'out_dim': 1,       # Output dimension
    'n_layers': 2,  # 3 also works well          # Number of layers
    'l_act': None,
}

default_mod_p = {
    'bs': 25,           # Size of the batch
    'lr': 5e-3,         # Learning rate
    'epochs': 200,  # 50,       # Number of training epochs 
    'pat': 25,  # 15,          # Number of non-decreasing epoch to stop training
    'wd': 1e-4,         # Weight decay
}



def get_signals(d_p, GSOs, get_srcs=False):
    range_GSO = np.arange(d_p['min_GSO'], d_p['max_GSO'])
    gsos_idx = np.random.choice(range_GSO, size=d_p['n_GSOs'], replace=False)
    sel_GSOs = GSOs[gsos_idx]
    signals_t, srcs_t, _ = dagu.create_diff_data(d_p['M'], sel_GSOs, d_p['max_src_node'], d_p['n_p_x'],
                                                 d_p['n_p_y'], d_p['n_sources'], src_t=d_p['src_t'],
                                                 torch_tensor=True, mask_sources=True)
    labels_t = torch.argmax(torch.abs(srcs_t), axis=1)

    signals = {'train': signals_t[:d_p['M_train']], 'val': signals_t[d_p['M_train']:-d_p['M_test']],
                 'test': signals_t[-d_p['M_test']:]}
    labels = {'train': labels_t[:d_p['M_train']], 'val': labels_t[d_p['M_train']:-d_p['M_test']],
              'test': labels_t[-d_p['M_test']:]}
    
    if get_srcs:
        srcs = {'train': srcs_t[:d_p['M_train']], 'val': srcs_t[d_p['M_train']:-d_p['M_test']],
              'test': srcs_t[-d_p['M_test']:]}
        
        return signals, labels, sel_GSOs, srcs
    
    return signals, labels, sel_GSOs


def run_exp(d_p, d_arc_args, d_mod_p, exps, verb=False):
    acc = np.zeros((d_p['n_tries'], len(exps)))
    times = np.zeros((d_p['n_tries'], len(exps)))

    t_begin = time.time()
    with tqdm(total=d_p['n_tries']*len(exps), disable=False) as pbar:
        for i in range(d_p['n_tries']):
            Adj, W, GSOs, Psi = utils.get_graph_data(d_p, get_Psi=True)
            signals, labels, sel_GSOs, srcs = get_signals(d_p, GSOs, get_srcs=True)

            for j, exp in enumerate(exps):
                # Combine default and experiment parameters    
                arc_p = {**exp['arc_p']}
                arc_p['args'] = {**d_arc_args, **arc_p['args']} if 'args' in arc_p.keys() else {**d_arc_args}
                mod_p = {**d_mod_p, **exp['mod_p']} if 'mod_p' in exp.keys() else d_mod_p

                if arc_p['arch'] == LinDAGClassModel:
                    # Combine default and experiment parameters    
                    if 'transp' in arc_p.keys() and arc_p['transp']:
                        dag_T = nx.from_numpy_array(Adj, create_using=nx.DiGraph())
                        Psi = np.array([dagu.compute_Dq(dag_T, i, d_p['N']) for i in range(d_p['N'])]).T
                        arc_p['transp'] = False

                    Psi_sel = utils.select_GSO(arc_p, Psi.T, None, W, Adj).numpy().T
                    lin_model = LinDAGClassModel(W, Psi_sel)
                    t_i = time.time()
                    # lin_model.fit(signals['train'], labels['train'])
                    lin_model.fit(signals['train'], srcs['train'])
                    t_e = time.time() - t_i
                    acc[i,j] = lin_model.test(signals['test'], labels['test'])
                
                elif arc_p['arch'] == DAGNN:
                    GSO = utils.select_GSO(arc_p, GSOs, sel_GSOs, W, Adj)
                    K = GSO.shape[0] if isinstance(GSO, torch.Tensor) and len(GSO.shape) == 3 else 0 
                    arch = utils.instantiate_arch(arc_p, K)
                    model = SrcIdModel(arch, device=device)
                    signals1, labels1 = DAGNN_model(Adj, signals, labels)
                    t_i = time.time()
                    model.fit(signals1, labels1, GSO, mod_p['lr'], mod_p['epochs'], mod_p['bs'], mod_p['wd'], patience=mod_p['pat'])
                    t_e = time.time() - t_i
                    acc[i,j] = model.test(signals1['test'], labels1['test'], GSO)

                elif arc_p['arch'] == DVAE:
                    GSO = utils.select_GSO(arc_p, GSOs, sel_GSOs, W, Adj)
                    K = GSO.shape[0] if isinstance(GSO, torch.Tensor) and len(GSO.shape) == 3 else 0 
                    arch = utils.instantiate_arch(arc_p, K)
                    model = SrcIdModel(arch, device=device)
                    signals1, labels1 = DVAE_model(Adj, signals, labels)
                    t_i = time.time()
                    model.fit(signals1, labels1, GSO, mod_p['lr'], mod_p['epochs'], mod_p['bs'], mod_p['wd'], patience=mod_p['pat'])
                    t_e = time.time() - t_i
                    acc[i,j] = model.test(signals1['test'], labels1['test'], GSO)

                else:
                    # Fit and test nonlinear models
                    GSO = utils.select_GSO(arc_p, GSOs, sel_GSOs, W, Adj)
                    K = GSO.shape[0] if isinstance(GSO, torch.Tensor) and len(GSO.shape) == 3 else 0 
                    arch = utils.instantiate_arch(arc_p, K)
                    model = SrcIdModel(arch, device=device)
                    t_i = time.time()
                    model.fit(signals, labels, GSO, mod_p['lr'], mod_p['epochs'], mod_p['bs'], mod_p['wd'], patience=mod_p['pat'])
                    t_e = time.time() - t_i
                    acc[i,j] = model.test(signals['test'], labels['test'], GSO)
                times[i,j] = t_e

                # Progress
                pbar.update(1)
                if verb:
                    print(f'-{i}. {exp["leg"]}: acc: {acc[i,j]:.3f} - time: {times[i,j]:.1f}')
    
    total_t = (time.time() - t_begin)/60
    print(f'----- Ellapsed time: {total_t:.2f} minutes -----')
    return acc, times


def run_exps(exps, d_arc_args, d_mod_p, d_dat_p, GSOs, W, Adj, pbar, verb=True, exp_desc='default'):
    # Create error variables
    acc_exps = np.zeros(len(exps))
    times_exps = np.zeros(len(exps))
    
    common_signals = all('dat_p' not in exp for exp in exps)
    if common_signals:
        signals, labels, sel_GSOs = get_signals(d_dat_p, GSOs)

    # signals, labels, sel_GSOs = get_signals(d_dat_p, GSOs)
    
    for k, exp in enumerate(exps):
        # Combine default and experiment parameters
        arc_p = {**exp['arc_p']}
        arc_p['args'] = {**d_arc_args, **arc_p['args']} if 'args' in arc_p.keys() else {**d_arc_args}
        mod_p = {**d_mod_p, **exp['mod_p']} if 'mod_p' in exp else d_mod_p

        if not common_signals:
            d_dat_p = {**d_dat_p, **exp['dat_p']}
            signals, labels, sel_GSOs = get_signals(d_dat_p, GSOs)

        GSO = utils.select_GSO(arc_p, GSOs, sel_GSOs, W, Adj)
        K = GSO.shape[0] if isinstance(GSO, torch.Tensor) and len(GSO.shape) == 3 else 0 
        arch = utils.instantiate_arch(arc_p, K)
        model = SrcIdModel(arch, device=device)
        t_i = time.time()
        model.fit(signals, labels, GSO, mod_p['lr'], mod_p['epochs'], mod_p['bs'], mod_p['wd'],
                  patience=mod_p['pat'])
        t_e = time.time() - t_i
        
        acc_exps[k] = model.test(signals['test'], labels['test'], GSO)
        times_exps[k] = t_e

        # Progress
        pbar.update(1)
        if verb:
            print(f'\t-{exp_desc}. {exp["leg"]}: acc: {acc_exps[k]:.3f} - time: {times_exps[k]:.1f}')

    return acc_exps, times_exps



mod_p_init = default_mod_p.copy()
mod_p_init['pat'] = 50  # 100
verb = True

Exps = [
    # Our Models
    # {'arc_p': {'arch': FB_DAGConv, 'GSO': 'GSOs'}, 'leg': 'DCN'},
    # {'arc_p': {'arch': FB_DAGConv, 'GSO': 'rnd_GSOs', 'n_gsos': 30}, 'leg': 'DCN-30'},
    # {'arc_p': {'arch': FB_DAGConv, 'GSO': 'rnd_GSOs', 'n_gsos': 10}, 'leg': 'DCN-10'},
    # {'arc_p': {'arch': DVAE, 'GSO': 'GSOs','max_n':10, 'nvt':1, 'START_TYPE':0, 'END_TYPE':1, 'hs':501,'nz':10, 'bidirectional': True, 'vid': True}, 'leg': 'DVAE'},
    {'arc_p': {'arch': SMLP , 'GSO': 'GSOs','transp': True, 'in_dim': 1, 'hid_dim': [1024], 'out_dim': 1, 'bias' : True }, 'leg': 'SMLP-1024'},
    # {'arc_p': {'arch': SMLP , 'GSO': 'GSOs','transp': True, 'in_dim': 1, 'hid_dim': [2048], 'out_dim': 1, 'bias' : True }, 'leg': 'SMLP-2048'},


    # {'arc_p': {'arch': DAGNN, 'GSO': 'GSOs', 'emb_dim': 1, 'hidden_dim':100, 'out_dim': 100, 'max_n': 100,'nvt':1 ,
    #            'START_TYPE': 0, 'END_TYPE': 1, 'hs':501, 'nz': 56, 'agg': "attn_h", 'num_layers':3, 'bidirectional': True, 'out_wx': False, 'out_pool_all': False, 
    #            'out_pool': P_MAX, 'dropout': 0.2, 'num_nodes': 100}, 'leg': 'DAGNN'},


    # {'arc_p': {'arch': DAGConv, 'GSO': 'GSOs'}, 'leg': 'DAGConv'},
    # {'arc_p': {'arch': DAGConv, 'GSO': 'rnd_GSOs', 'n_gsos': 30}, 'leg': 'DAGConv-30'},

    # {'arc_p': {'arch': ADCN, 'GSO': 'GSOs', 'args': {'mlp_layers': 4, 'n_layers': 4}}, 'leg': 'ADCN-4'},
    # {'arc_p': {'arch': ADCN, 'GSO': 'GSOs', 'args': {'mlp_layers': 5, 'n_layers': 4}}, 'leg': 'ADCN-5'},

    # Our Models - Transposed
    {'arc_p': {'arch': FB_DAGConv, 'GSO': 'GSOs', 'transp': True}, 'leg': 'DCN-T'},
    # {'arc_p': {'arch': FB_DAGConv, 'GSO': 'rnd_GSOs', 'n_gsos': 30, 'transp': True}, 'leg': 'DCN-30-T'},
    # {'arc_p': {'arch': FB_DAGConv, 'GSO': 'rnd_GSOs', 'n_gsos': 10, 'transp': True}, 'leg': 'DCN-10-T'},
    # {'arc_p': {'arch': FB_DAGConv, 'GSO': 'rnd_GSOs', 'n_gsos': 5, 'transp': True}, 'leg': 'DCN-5-T'},

    # {'arc_p': {'arch': DAGConv, 'GSO': 'GSOs', 'transp': True}, 'leg': 'DAGConv-T'},
    # {'arc_p': {'arch': DAGConv, 'GSO': 'rnd_GSOs', 'n_gsos': 30, 'transp': True}, 'leg': 'DAGConv-30-T'},

    
    # {'arc_p': {'arch': ADCN, 'GSO': 'GSOs', 'transp': True, 'args': {'mlp_layers': 4, 'n_layers': 2,}}, 'leg': 'ADCN-4-2-T'},
    # {'arc_p': {'arch': ADCN, 'GSO': 'GSOs', 'transp': True, 'args': {'mlp_layers': 5, 'n_layers': 2,}}, 'leg': 'ADCN-5-2-T'},
    # {'arc_p': {'arch': ADCN, 'GSO': 'GSOs', 'transp': True, 'args': {'mlp_layers': 4, 'n_layers': 4,}}, 'leg': 'ADCN-4-4-T'},
    # {'arc_p': {'arch': ADCN, 'GSO': 'GSOs', 'transp': True, 'args': {'mlp_layers': 5, 'n_layers': 4,}}, 'leg': 'ADCN-5-4-T'},


    # Linear baselines
    # {'arc_p': {'arch': LinDAGClassModel, 'GSO': 'GSOs'}, 'leg': 'Linear'},
    # {'arc_p': {'arch': LinDAGClassModel, 'GSO': 'rnd_GSOs', 'n_gsos': 30}, 'leg': 'Linear-30'},
    # {'arc_p': {'arch': LinDAGClassModel, 'GSO': 'rnd_GSOs', 'n_gsos': 10}, 'leg': 'Linear-10'},
    # {'arc_p': {'arch': LinDAGClassModel, 'GSO': 'GSOs', 'transp': True}, 'leg': 'Linear-T'},
    # {'arc_p': {'arch': LinDAGClassModel, 'GSO': 'rnd_GSOs', 'n_gsos': 30, 'transp': True}, 'leg': 'Linear-30-T'},
    # {'arc_p': {'arch': LinDAGClassModel, 'GSO': 'rnd_GSOs', 'n_gsos': 10, 'transp': True}, 'leg': 'Linear-10-T'},

    # GNN Baselines
    # {'arc_p': {'arch': FB_DAGConv, 'GSO': 'A_pows', 'K': 2, 'transp': True}, 'leg': 'FB-GCNN-2'},
    # {'arc_p': {'arch': FB_DAGConv, 'GSO': 'A_pows', 'K': 3, 'transp': True}, 'leg': 'FB-GCNN-3'},
    # {'arc_p': {'arch': FB_DAGConv, 'GSO': 'A_pows', 'K': 4, 'transp': True}, 'leg': 'FB-GCNN-4'},
    # {'arc_p': {'arch': MyGCNN, 'GSO': 'A', 'transp': True}, 'leg': 'GNN-A'},
    # {'arc_p': {'arch': GAT, 'GSO': 'A-dgl', 'transp': True, 'args': {'num_heads': 2, 'hid_dim': 16,
    #  'gat_params': {'attn_drop': 0}}}, 'leg': 'GAT'},
    # {'arc_p': {'arch': GraphSAGE, 'GSO': 'A-dgl', 'transp': True, 'args': {'aggregator': 'mean'}}, 'leg': 'GraphSAGE-A'},
    # {'arc_p': {'arch': GIN, 'GSO': 'A-dgl', 'transp': True, 'args': {'aggregator': 'sum'}}, 'leg': 'GIN-A'},
    # {'arc_p': {'arch': GIN, 'GSO': 'A-dgl', 'transp': True, 'args': {'aggregator': 'sum', 'mlp_layers': 4}}, 'leg': 'GIN-A'},
    # {'arc_p': {'arch': MLP, 'GSO': None}, 'leg': 'MLP'},
    # {'arc_p': {'arch': MLP, 'GSO': None, 'args': {'n_layers': 4}}, 'leg': 'MLP-4'},
    ]

acc, times = run_exp(data_p, default_arch_args, mod_p_init, Exps, verb=verb)

import os
import sys

import torch
import torch.nn as nn

# Allow running from the tools/ directory while importing the project package.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from src.arch import FB_DAGConv, ADCN
import src.utils as utils

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

default_arch_args = {
    'in_dim': 1,
    'hid_dim': 32,
    'out_dim': 1,
    'n_layers': 2,
    'l_act': None,
}

# Values from source_id.ipynb
configs = [
    # DCN variants
    {'name': 'DCN (K=25)', 'arch': FB_DAGConv, 'K': 25, 'args': {}},
    {'name': 'DCN-30', 'arch': FB_DAGConv, 'K': 30, 'args': {}},
    {'name': 'DCN-10', 'arch': FB_DAGConv, 'K': 10, 'args': {}},
    {'name': 'DCN-5', 'arch': FB_DAGConv, 'K': 5, 'args': {}},
    
    # ADCN variants
    # ADCN uses default filter_coefs=torch.ones(1), so K=1 effectively for params, but we should verify.
    # The GSO argument just changes the input GSOs but not the model params if filter_coefs is fixed.
    {'name': 'ADCN-4 (MLP-4, L-4)', 'arch': ADCN, 'K': 1, 'args': {'mlp_layers': 4, 'n_layers': 4}},
    {'name': 'ADCN-5 (MLP-5, L-4)', 'arch': ADCN, 'K': 1, 'args': {'mlp_layers': 5, 'n_layers': 4}},
    
    {'name': 'ADCN-4-2 (MLP-4, L-2)', 'arch': ADCN, 'K': 1, 'args': {'mlp_layers': 4, 'n_layers': 2}},
    {'name': 'ADCN-5-2 (MLP-5, L-2)', 'arch': ADCN, 'K': 1, 'args': {'mlp_layers': 5, 'n_layers': 2}},
    {'name': 'ADCN-4-4 (MLP-4, L-4)', 'arch': ADCN, 'K': 1, 'args': {'mlp_layers': 4, 'n_layers': 4}}, 
    {'name': 'ADCN-5-4 (MLP-5, L-4)', 'arch': ADCN, 'K': 1, 'args': {'mlp_layers': 5, 'n_layers': 4}}, 
]

print(f"{'Model':<30} | {'Parameters':<10}")
print("-" * 43)

for config in configs:
    # Merge args
    args = default_arch_args.copy()
    args.update(config['args'])
    
    try:
        if config['arch'] == ADCN:
            # ADCN constructor: 
            # (in_dim, hid_dim, out_dim, n_layers, mlp_layers=2, filter_coefs=torch.ones(1), ...)
            # We assume filter_coefs default (ones(1)) as in Exps
            
            model = ADCN(
                in_dim=args['in_dim'],
                hid_dim=args['hid_dim'],
                out_dim=args['out_dim'],
                n_layers=args['n_layers'],
                mlp_layers=args.get('mlp_layers', 2)
            )
        else:
            # FB_DAGConv constructor:
            # (in_dim, hid_dim, out_dim, K, n_layers, ...)
            model = config['arch'](
                in_dim=args['in_dim'],
                hid_dim=args['hid_dim'],  # Note: FB_DAGConv uses hid_dim logic 
                out_dim=args['out_dim'],
                K=config['K'],
                n_layers=args['n_layers']
            )
            
        n_params = count_params(model)
        print(f"{config['name']:<30} | {n_params:<10}")
        
    except Exception as e:
        print(f"{config['name']:<30} | Error: {e}")
        import traceback
        traceback.print_exc()

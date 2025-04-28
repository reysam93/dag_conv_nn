from dgl.nn import GATConv, GraphConv, SAGEConv, GINConv
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineArch(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers=2, act=F.relu, l_act=None,
                 bias=True, dropout=0):
        super(BaselineArch, self).__init__()
        self.in_d = in_dim
        self.hid_d = hid_dim
        self.out_d = out_dim
        self.dropout = nn.Dropout(p=dropout)
        self.act = act
        self.l_act = l_act

        self.convs = self._create_conv_layers(n_layers, bias)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, X, A):
        for _, conv in enumerate(self.convs[:-1]):
            X = self.act(conv(A, X))
            X = self.dropout(X)

        X_out = self.convs[-1](A, X)
        return self.l_act(X_out) if self.l_act else X_out


class MyGCNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, bias):
        super(MyGCNNLayer, self).__init__()
        self.in_d = in_dim
        self.out_d = out_dim
        self._init_parameters(bias)

    def _init_parameters(self, bias):
        self.W = nn.Parameter(torch.empty((self.in_d, self.out_d)))
        nn.init.xavier_uniform_(self.W)

        if bias:
            self.b = nn.Parameter(torch.empty(self.out_d))
            nn.init.constant_(self.b.data, 0.)
        else:
            self.b = None

    def forward(self, A, X):
        X_out = A @ (X @ self.W)

        if self.b is not None:
            return X_out + self.b[None,:]
        else:
            return X_out


class MyGCNN(BaselineArch):
    def _create_conv_layers(self, n_layers: int, bias: bool) -> nn.ModuleList:
        convs = nn.ModuleList()
        
        if n_layers > 1:
            convs.append(MyGCNNLayer(self.in_d, self.hid_d, bias))
            for _ in range(n_layers - 2):
                convs.append(MyGCNNLayer(self.hid_d, self.hid_d, bias))
            convs.append(MyGCNNLayer(self.hid_d, self.out_d, bias))
        else:
            convs.append(MyGCNNLayer(self.in_d, self.out_d, bias))

        return convs


class GraphSAGE(BaselineArch):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers=2, act=F.relu, l_act=None,
                 bias=True, dropout=0, aggregator='mean'):
        self.aggregator = aggregator
        super(GraphSAGE, self).__init__(in_dim, hid_dim, out_dim, n_layers, act, l_act,
                                        bias, dropout)
        

    def _create_conv_layers(self, n_layers: int, bias: bool) -> nn.ModuleList:
        convs = nn.ModuleList()

        if n_layers > 1:
            convs.append(SAGEConv(self.in_d, self.hid_d, self.aggregator, bias=bias))
            for _ in range(n_layers - 2):
                convs.append(SAGEConv(self.hid_d, self.hid_d, self.aggregator, bias=bias))
            convs.append(SAGEConv(self.hid_d, self.out_d, self.aggregator, bias=bias))
        else:
            convs.append(SAGEConv(self.in_d, self.out_d, self.aggregator, bias=bias))

        return convs

    def forward(self, h, graph):
        h = h.transpose(0,1)
        return super().forward(h, graph).transpose(1, 0)


class GIN(BaselineArch):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers=2, act=F.relu, l_act=None,
                 bias=True, dropout=0, aggregator='sum', mlp_layers = 2):
        self.aggregator = aggregator
        self.apply_func = MLP
        self.mlp_layers = mlp_layers
        super(GIN, self).__init__(in_dim, hid_dim, out_dim, n_layers, act, l_act,
                                  bias, dropout)
    
    def _create_conv_layers(self, n_layers: int, bias: bool) -> nn.ModuleList:
        convs = nn.ModuleList()

        # Last actication of apply_func is always set to None because the non-linearity is applyed in
        # the forward pass of the BaselineArch class
        if n_layers > 1:
            apply_func = self.apply_func(self.in_d, self.hid_d, self.hid_d, bias=bias, act=self.act,
                                         l_act=None, n_layers=self.mlp_layers)
            convs.append(GINConv(apply_func, self.aggregator))
            for _ in range(n_layers - 2):
                apply_func = self.apply_func(self.hid_d, self.hid_d, self.hid_d, bias=bias, act=self.act,
                                             l_act=None, n_layers=self.mlp_layers)
                convs.append(GINConv(apply_func, self.aggregator))
            apply_func = self.apply_func(self.hid_d, self.hid_d, self.out_d, bias=bias, act=self.act,
                                         l_act=None, n_layers=self.mlp_layers)
            convs.append(GINConv(apply_func, self.aggregator))
        else:
            apply_func = self.apply_func(self.in_d, self.hid_d, self.out_d, bias=bias, act=self.act,
                                         l_act=None, n_layers=self.mlp_layers)
            convs.append(GINConv(apply_func, self.aggregator))

        return convs

    def forward(self, h, graph):
        h = h.transpose(0,1)
        return super().forward(h, graph).transpose(1, 0)


class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers=2, dropout=0., bias=True,
                 act=F.relu, l_act=None):
        super(MLP, self).__init__()
        self.in_d = in_dim
        self.hid_d = hid_dim
        self.out_d = out_dim
        self.act = act
        self.l_act = l_act
        self.dropout = nn.Dropout(p=dropout)

        self.lin_layers = self._create_lin_layers(n_layers, bias)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _create_lin_layers(self, n_layers: int, bias: bool) -> nn.ModuleList:
        lin_layers = nn.ModuleList()

        if n_layers > 1:
            lin_layers.append(nn.Linear(self.in_d, self.hid_d, bias=bias))
            for _ in range(n_layers - 2):
                lin_layers.append(nn.Linear(self.hid_d, self.hid_d, bias=bias))
            lin_layers.append(nn.Linear(self.hid_d, self.out_d, bias=bias))
        else:
            lin_layers.append(nn.Linear(self.in_d, self.out_d, bias=bias))

        return lin_layers

    def forward(self, h):
        for _, lin_layer in enumerate(self.lin_layers[:-1]):
            h = self.act(lin_layer(h))
            h = self.dropout(h)

        h = self.lin_layers[-1](h)
        return self.l_act(h) if self.l_act else h


class GAT(nn.Module):
    """
    Graph Attention Network Class
    """
    def __init__(self, in_dim, hid_dim, out_dim, num_heads, gat_params,
                 act=F.elu, l_act=None, n_layers=None):
        super(GAT, self).__init__()

        if n_layers is not None:
            print('WARNING: GAT is implemeted with a fixed number of layers. The argument is ignored.')

        self.layer1 = GATConv(in_dim, hid_dim, num_heads, **gat_params)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.layer2 = GATConv(hid_dim * num_heads, out_dim, 1, **gat_params)
        self.act = act
        self.l_act = l_act

    def forward(self, h, graph):
        h = h.transpose(0, 1)
        h = self.layer1(graph, h)
        h = h.flatten(2)        
        h = self.act(h)
        h = self.layer2(graph, h).squeeze(-1).transpose(0, 1)
        return self.l_act(h) if self.l_act else h


from dgl.nn import GATConv, GraphConv
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers=2, dropout=0., bias=True,
                 act=F.relu, last_act=None,):
        super(MLP, self).__init__()
        self.in_d = in_dim
        self.hid_d = hidden_dim
        self.out_d = out_dim
        self.act = act
        self.l_act = last_act
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
    def __init__(self, in_dim, hidden_dim, out_dim, graph, num_heads, gat_params,
                 act=F.elu, last_act=None):
        super(GAT, self).__init__()
        self.graph = graph
        self.layer1 = GATConv(in_dim, hidden_dim, num_heads, **gat_params)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.layer2 = GATConv(hidden_dim * num_heads, out_dim, 1, **gat_params)
        self.act = act
        self.l_act = last_act

    def forward(self, h):
        if len(h.shape) == 3:
            h = h.squeeze().T
        h = self.layer1(self.graph, h)
        # concatenate
        h = h.flatten(1)
        h = self.act(h)
        h = self.layer2(self.graph, h).squeeze()
        return self.l_act(h) if self.l_act else h
    

class GCNN_2L(nn.Module):
    """
    2-layer Graph Convolutional Neural Network Class as in Kipf
    """
    def __init__(self, in_dim, hidden_dim, out_dim, graph, act=F.relu, last_act=None,
                 norm='both', bias=True, dropout=0):
        super(GCNN_2L, self).__init__()
        self.graph = graph
        self.layer1 = GraphConv(in_dim, hidden_dim, bias=bias, norm=norm)
        self.layer2 = GraphConv(hidden_dim, out_dim, bias=bias, norm=norm)
        self.dropout = nn.Dropout(p=dropout)
        self.act = act
        self.l_act = last_act

    def forward(self, h):
        h = h.transpose(0,1)
        h = self.layer1(self.graph, h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.layer2(self.graph, h).transpose(1, 0)
        return self.l_act(h) if self.l_act else h
    

class GCNN(nn.Module):
    """
    Graph Convolutional Neural Network Class as in Kipf
    """
    def __init__(self, in_dim, hidden_dim, out_dim, graph, n_layers=2, act=F.relu,
                 last_act=None, norm='both', bias=True, dropout=0):
        super(GCNN, self).__init__()
                
        self.in_d = in_dim
        self.hid_d = hidden_dim
        self.out_d = out_dim
        self.graph = graph
        self.dropout = nn.Dropout(p=dropout)
        self.n_layers = n_layers
        self.act = act
        self.l_act = last_act
        self.convs = self._create_conv_layers(n_layers, bias, norm)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)


    def _create_conv_layers(self, n_layers: int, bias: bool, norm: str) -> nn.ModuleList:
        convs = nn.ModuleList()

        if n_layers > 1:
            convs.append(GraphConv(self.in_d, self.hid_d, bias=bias, norm=norm))
            for _ in range(n_layers - 2):
                convs.append(GraphConv(self.hid_d, self.hid_d, bias=bias, norm=norm))
            convs.append(GraphConv(self.hid_d, self.out_d, bias=bias, norm=norm))
        else:
            convs.append(GraphConv(self.in_d, self.out_d, bias=bias, norm=norm))

        return convs

    def forward(self, h):
        h = h.transpose(0,1)
        for _, conv in enumerate(self.convs[:-1]):
            h = self.act(conv(self.graph, h))
            h = self.dropout(h)

        h = self.convs[-1](self.graph, h).transpose(1, 0)
        return self.l_act(h) if self.l_act else h

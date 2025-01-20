import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch import Tensor
import torch_geometric as tg
from torch_geometric.typing import OptTensor
from torch_geometric.utils import softmax
# from torch_scatter import scatter_add
from torch_geometric.nn.glob import *
from torch_geometric.nn import MessagePassing
import sys
import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src_DAGNN'))
from src.utils_Dagnn import *
from src.batch import Batch
import copy



class DVAE_PYG(nn.Module):
    def __init__(self, max_n, nvt, START_TYPE, END_TYPE, hs=501, nz=56, bidirectional=False, vid=True, num_layers=1):
        super(DVAE_PYG, self).__init__()
        self.max_n = max_n  # maximum number of vertices
        self.nvt = nvt  # number of vertex types
        self.START_TYPE = START_TYPE
        self.END_TYPE = END_TYPE
        self.hs = hs  # hidden state size of each vertex
        self.nz = nz  # size of latent representation z
        self.gs = hs  # size of graph state
        self.bidir = bidirectional  # whether to use bidirectional encoding
        self.vid = vid
        self.device = None
        self.regression_layer = nn.Linear(self.hs, self.max_n)


        if self.vid:
            self.vs = hs + max_n  # vertex state size = hidden state + vid
        else:
            self.vs = hs
        self.num_layers = num_layers
        # 0. encoding-related
        self.grue_forward = nn.ModuleList([nn.GRUCell(nvt, hs) if l == 0 else nn.GRUCell(hs, hs) for l in range(num_layers)])  # encoder GRU
        self.grue_backward = nn.ModuleList([nn.GRUCell(nvt, hs) if l == 0 else nn.GRUCell(hs, hs) for l in range(num_layers)]) # backward encoder GRU
        self.fc1 = nn.Linear(self.gs, nz)  # latent mean
        self.fc2 = nn.Linear(self.gs, nz)  # latent logvar
            
        # 1. decoding-related
        self.grud = nn.ModuleList([nn.GRUCell(nvt, hs) if l == 0 else nn.GRUCell(hs, hs) for l in range(num_layers)]) # decoder GRU  # TODO we here leave one layer?
        self.fc3 = nn.Linear(nz, hs)  # from latent z to initial hidden state h0
        self.add_vertex = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.ReLU(),
                nn.Linear(hs * 2, nvt)
                )  # which type of new vertex to add f(h0, hg)
        self.add_edge = nn.Sequential(
                nn.Linear(hs * 2, hs * 4), 
                nn.ReLU(), 
                nn.Linear(hs * 4, 1)
                )  # whether to add edge between v_i and v_new, f(hvi, hnew)
        # print("sizes", self.vs, hs, nvt)
        # 2. gate-related
        self.gate_forward = nn.ModuleList([nn.Sequential(
                nn.Linear(self.vs, hs), 
                nn.Sigmoid()
                )for _ in range(num_layers)])
        self.gate_backward = nn.ModuleList([nn.Sequential(
                nn.Linear(self.vs, hs), 
                nn.Sigmoid()
                )for _ in range(num_layers)])
        self.mapper_forward = nn.ModuleList([nn.Sequential(
                nn.Linear(self.vs, hs, bias=False),
                ) for _ in range(num_layers)])  # disable bias to ensure padded zeros also mapped to zeros
        self.mapper_backward = nn.ModuleList([nn.Sequential(
                nn.Linear(self.vs, hs, bias=False), 
                ) for _ in range(num_layers)])

        # 3. bidir-related, to unify sizes
        if self.bidir:
            self.hv_unify = nn.Sequential(
                    nn.Linear(hs * 2, hs), 
                    )
            self.hg_unify = nn.Sequential(
                    nn.Linear(self.gs * 2 * num_layers, self.gs), # VT skip conn
                    )

        # 4. other
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.logsoftmax1 = nn.LogSoftmax(1)

    def get_device(self):
        if self.device is None:
            self.device = next(self.parameters()).device
        return self.device
    
    def _get_zeros(self, n, length):
        return torch.zeros(n, length).to(self.get_device()) # get a zero hidden state

    def _get_zero_hidden(self, n=1):
        return self._get_zeros(n, self.hs) # get a zero hidden state

    def _one_hot(self, idx, length):
        if type(idx) in [list, range]:
            if idx == []:
                return None
            idx = torch.LongTensor(idx).unsqueeze(0).t()
            x = torch.zeros((len(idx), length)).scatter_(1, idx, 1).to(self.get_device())
        else:
            idx = torch.LongTensor([idx]).unsqueeze(0)
            x = torch.zeros((1, length)).scatter_(1, idx, 1).to(self.get_device())
        return x

    def _gated(self, h, gate, mapper):
        # print('gated')
        # print(h)
        return gate(h) * mapper(h)

    def _collate_fn(self, G):
        return [copy.deepcopy(g) for g in G]

    def _propagate_to(self, G, v, propagator, H=None, reverse=False):
        # propagate messages to vertex index v for all graphs in G
        # return the new messages (states) at v
        G = [g.to(self.get_device()) for g in G if g.x.shape[0] > v]
        if len(G) == 0:
            return
        if H is not None:
            idx = [i for i, g in enumerate(G) if g.x.shape[0] > v]
            H = H[idx]
        # v_types = [g.vs[v]['type'] for g in G]
        # X = self._one_hot(v_types, self.nvt)
        X = torch.stack([g.x[v] for g in G], dim=0)
        
        Hv=X
        for l in range(self.num_layers):
            istr = str(l)
            if reverse:
                H_name = 'H_backward'+istr # name of the hidden states attribute
                #  H_pred = [[g.vs[x][H_name] for x in g.successors(v)] for g in G]
                #             if self.vid:
                #                 vids = [self._one_hot(g.successors(v), self.max_n) for g in G]
                H_pred = []
                vids = []
                for g in G:
                    # np_idx = g.bi_node_parent_index[1][0] == v
                    # np_idx = g.bi_node_parent_index[1][1][np_idx]
                    np_idx = g.edge_index[0] == v
                    np_idx = g.edge_index[1][np_idx]
                    print(np_idx)
                    H_pred += [[g.vs[x][H_name] for x in np_idx]]
                    if H_pred[0] and self.vid:
                        vids += [self._one_hot(np_idx.tolist(), self.max_n)]
                gate, mapper = self.gate_backward, self.mapper_backward
            else:
                H_name = 'H_forward'+istr  # name of the hidden states attribute
                # H_pred = [[g.vs[x][H_name] for x in g.predecessors(v)] for g in G]
                # if self.vid:
                #     vids = [self._one_hot(g.predecessors(v), self.max_n) for g in G]
                H_pred = []
                vids = []
                for g in G:
                    # np_idx = g.bi_node_parent_index[0][0] == v
                    # np_idx = g.bi_node_parent_index[0][1][np_idx]
                    np_idx = g.edge_index[1] == v
                    np_idx = g.edge_index[0][np_idx]
                    H_pred += [[g.vs[x][H_name] for x in np_idx]]
                    if H_pred[0] and self.vid:
                        vids += [self._one_hot(np_idx.tolist(), self.max_n)]
                gate, mapper = self.gate_forward, self.mapper_forward
            if H_pred[0] and self.vid:  #H_pred and H_pred[0] and
                H_pred = [[torch.cat([x[i], y[i:i+1]], 1) for i in range(len(x))] for x, y in zip(H_pred, vids)]
            # if h is not provided, use gated sum of v's predecessors' states as the input hidden state
            if H is None:
                max_n_pred = max([len(x) for x in H_pred])  # maximum number of predecessors
                if max_n_pred == 0:
                    H = self._get_zero_hidden(len(G))
                else:
                    # H_pred = [torch.cat(h_pred +
                    #             [self._get_zeros(max_n_pred - len(h_pred), self.vs)], 0).unsqueeze(0)
                    #             for h_pred in H_pred]  # pad all to same length
                    H_pred = [torch.cat(h_pred +[self._get_zeros(max_n_pred - len(h_pred), self.vs)], 0).unsqueeze(0)
                              for h_pred in H_pred]
                    H_pred = torch.cat(H_pred, 0)  # batch * max_n_pred * vs
                    H = self._gated(H_pred, gate[l], mapper[l]).sum(1)  # batch * hs
            # print(H)
            Hv = propagator[l](Hv, H)
            for i, g in enumerate(G):
                g.vs[v][H_name] = Hv[i:i+1]
            # print(Hv)
        return Hv

    def _ipropagate_to(self, G, v, propagator, H=None, reverse=False):
        # propagate messages to vertex index v for all graphs in G
        # return the new messages (states) at v
        G = [g for g in G if g.vcount() > v]
        if len(G) == 0:
            return
        if H is not None:
            idx = [i for i, g in enumerate(G) if g.vcount() > v]
            H = H[idx]
        v_types = [g.vs[v]['type'] for g in G]
        X = self._one_hot(v_types, self.nvt)
        if reverse:
            H_name = 'H_backward'  # name of the hidden states attribute
            H_pred = [[g.vs[x][H_name] for x in g.successors(v)] for g in G]
            if self.vid:
                vids = [self._one_hot(g.successors(v), self.max_n) for g in G]
            gate, mapper = self.gate_backward[0], self.mapper_backward[0]
        else:
            H_name = 'H_forward'  # name of the hidden states attribute
            H_pred = [[g.vs[x][H_name] for x in g.predecessors(v)] for g in G]
            if self.vid:
                vids = [self._one_hot(g.predecessors(v), self.max_n) for g in G]
            gate, mapper = self.gate_forward[0], self.mapper_forward[0]
        if self.vid:
            H_pred = [[torch.cat([x[i], y[i:i + 1]], 1) for i in range(len(x))] for x, y in zip(H_pred, vids)]
        # if h is not provided, use gated sum of v's predecessors' states as the input hidden state
        if H is None:
            max_n_pred = max([len(x) for x in H_pred])  # maximum number of predecessors
            if max_n_pred == 0:
                H = self._get_zero_hidden(len(G))
            else:
                H_pred = [torch.cat(h_pred +
                                    [self._get_zeros(max_n_pred - len(h_pred), self.vs)], 0).unsqueeze(0)
                          for h_pred in H_pred]  # pad all to same length
                H_pred = torch.cat(H_pred, 0)  # batch * max_n_pred * vs
                H = self._gated(H_pred, gate, mapper).sum(1)  # batch * hs
        Hv = propagator[0](X, H)
        for i, g in enumerate(G):
            g.vs[v][H_name] = Hv[i:i + 1]
        return Hv

    def _propagate_from(self, G, v, propagator, H0=None, reverse=False):
        # perform a series of propagation_to steps starting from v following a topo order
        # assume the original vertex indices are in a topological order
        # prop_order = G.top_order.tolist()
        # if reverse:
        #     prop_order.reverse()
        if reverse:
            prop_order = range(v, -1, -1)
        else:
            prop_order = range(v, self.max_n)
        Hv = self._propagate_to(G, v, propagator, H0, reverse=reverse)  # the initial vertex
        for v_ in prop_order[1:]:
            self._propagate_to(G, v_, propagator, reverse=reverse)
        return Hv

    def _update_v(self, G, v, H0=None):
        # perform a forward propagation step at v when decoding to update v's state
        self._propagate_to(G, v, self.grud, H0, reverse=False)
        return

    def _update_iv(self, G, v, H0=None):
        # perform a forward propagation step at v when decoding to update v's state
        self._ipropagate_to(G, v, self.grud, H0, reverse=False)
        return

    def _get_vertex_state(self, G, v):
        # get the vertex states at v
        Hv = []
        istr=str(self.num_layers-1)
        for g in G:
            if v >= g.x.shape[0]:
                hv = self._get_zero_hidden()
            else:
                hv = g.vs[v]['H_forward'+istr]
            Hv.append(hv)
        Hv = torch.cat(Hv, 0)
        return Hv

    def _get_ivertex_state(self, G, v):
        # get the vertex states at v
        Hv = []
        for g in G:
            if v >= g.vcount():
                hv = self._get_zero_hidden()
            else:
                hv = g.vs[v]['H_forward'+str(self.num_layers - 1)]
            Hv.append(hv)
        Hv = torch.cat(Hv, 0)
        return Hv

    def _get_graph_state(self, G, decode=False):
        # get the graph states
        # when decoding, use the last generated vertex's state as the graph state
        # when encoding, use the ending vertex state or unify the starting and ending vertex states
        Hg = []
        istr=str(self.num_layers-1)
        for g in G:
            hg = g.vs[g.x.shape[0]-1]['H_forward'+istr]
            if self.bidir and not decode:  # decoding never uses backward propagation
                hg_b = g.vs[0]['H_backward'+istr]
                hg = torch.cat([hg, hg_b], 1)
            Hg.append(hg)
        Hg = torch.cat(Hg, 0)
        if self.bidir and not decode:
            Hg = self.hg_unify(Hg)
        return Hg

    def _get_igraph_state(self, G, decode=False):
        # get the graph states
        # when decoding, use the last generated vertex's state as the graph state
        # when encoding, use the ending vertex state or unify the starting and ending vertex states
        Hg = []
        istr = str(self.num_layers - 1)
        for g in G:
            hg = g.vs[g.vcount() - 1]['H_forward'+istr]
            if self.bidir and not decode:  # decoding never uses backward propagation
                hg_b = g.vs[0]['H_backward'+istr]
                hg = torch.cat([hg, hg_b], 1)
            Hg.append(hg)
        Hg = torch.cat(Hg, 0)
        if self.bidir and not decode:
            Hg = self.hg_unify(Hg)
        return Hg

    def encode(self, G):
        # encode graphs G into latent vectors
        if type(G) != list:
            G = [G]
        self._propagate_from(G, 0, self.grue_forward, H0=self._get_zero_hidden(len(G)),
                             reverse=False)
        if self.bidir:
            self._propagate_from(G, self.max_n-1, self.grue_backward, 
                                 H0=self._get_zero_hidden(len(G)), reverse=True)
        Hg = self._get_graph_state(G)
        # mu, logvar = self.fc1(Hg), self.fc2(Hg) 
        mu = self.regression_layer(Hg)
        return mu


    def forward(self, G):
        mu = self.encode(G)
        # loss, _, _ = self.loss(mu, logvar, G)
        return mu



class DAGNN(DVAE_PYG):                        

    def __init__(self, emb_dim, hidden_dim, out_dim,
                 max_n, nvt, START_TYPE, END_TYPE, hs, nz,
                 num_layers=2, bidirectional=False, agg=NA_ATTN_H, out_wx=False, out_pool_all=False, out_pool=P_MAX,
                 dropout=0.0, num_nodes=8):  # D-VAE SPECIFIC num_nodes
        super().__init__(max_n, nvt, START_TYPE, END_TYPE, hs, nz, bidirectional=bidirectional, num_layers=num_layers)

        self.num_nodes = num_nodes  # D-VAE SPECIFIC

        assert not ("_x" in agg and "attn" in agg) # decoding not adapted for this yet

        # configuration
        self.agg = agg
        self.agg_attn = "attn" in agg
        self.agg_attn_x = "_x" in agg
        self.bidirectional = bidirectional
        self.dirs = [0, 1] if bidirectional else [0]
        self.num_layers = num_layers
        self.out_wx = out_wx
        self.output_all = out_pool_all
        self.loss1 = torch.nn.MSELoss(reduction='mean')
        self.regression_layer = nn.Linear(self.hs, self.num_nodes)



        # dimensions
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.out_hidden_dim = emb_dim + self.hidden_dim * self.num_layers if out_wx else self.hidden_dim * self.num_layers  # D-VAE SPECIFIC, USING UNIFY, no *len(self.dirs)

        # aggregate
        num_rels = 1
        pred_dim = (self.hidden_dim + num_nodes)  # D-VAE SPECIFIC
        attn_dim = self.emb_dim if "_x" in agg else (self.hidden_dim + num_nodes)  # D-VAE SPECIFIC
        if "self_attn" in agg:
            # it wouldn't make sense to perform attention based on h when aggregating x... so no hidden_dim needed
            self.node_aggr_0 = nn.ModuleList([
                SelfAttnConv(attn_dim, num_relations=num_rels) for _ in range(num_layers)])
            self.node_aggr_1 = nn.ModuleList([
                SelfAttnConv(attn_dim, num_relations=num_rels, reverse=True) for _ in range(num_layers)])
        elif "attn" in agg:
            self.node_aggr_0 = nn.ModuleList([
                AttnConv(self.emb_dim if l == 0 else attn_dim, pred_dim, num_relations=num_rels, attn_dim=attn_dim) for l in range(num_layers)])
            self.node_aggr_1 = nn.ModuleList([
                AttnConv(self.emb_dim if l == 0 else attn_dim, pred_dim, num_relations=num_rels, attn_dim=attn_dim, reverse=True) for l in range(num_layers)])
        elif agg == NA_GATED_SUM:
            # D-VAE SPECIFIC, use super's layers since also used in decoding
            self.node_aggr_0 = nn.ModuleList([
                GatedSumConv(pred_dim, num_rels, mapper=self.mapper_forward[l], gate=self.gate_forward[l]) for l in range(num_layers)])
            self.node_aggr_1 = nn.ModuleList([
                GatedSumConv(pred_dim, num_rels, mapper=self.mapper_backward[l], gate=self.gate_backward[l], reverse=True) for l in range(num_layers)])
        else:
            node_aggr = AggConv(agg, num_rels, pred_dim)
            self.node_aggr_0 = nn.ModuleList([node_aggr for _ in range(num_layers)])  # just to have same format
            node_aggr = AggConv(agg, num_rels, pred_dim, reverse=True)
            self.node_aggr_1 = nn.ModuleList([node_aggr for _ in range(num_layers)])  # just to have same format

        # RNN
        self.__setattr__("cells_{}".format(0), self.grue_forward)
        if self.bidirectional:
            self.__setattr__("cells_{}".format(1), self.grue_backward)

        # readout
        self._readout = self._out_nodes_self_attn if out_pool == P_ATTN else getattr(tg.nn, 'global_{}_pool'.format(out_pool))

        # output
        self.dropout = nn.Dropout(dropout)

        self.out_linear = torch.nn.Linear(self.out_hidden_dim, out_dim) if self.num_layers > 1 else None

    def _out_nodes_self_attn(self, h, batch):
        attn_weights = self.self_attn_linear_out(h)
        attn_weights = F.softmax(attn_weights, dim=-1)
        return global_add_pool(attn_weights * h, batch)

    def _get_output_nodes(self, G):
        if self.bidirectional:
            layer0 = G.bi_layer_index[0][0] == 0
            layer0 = G.bi_layer_index[0][1][layer0]
            return torch.cat([G.h[G.target_index], G.h[layer0]], dim=0), \
                   torch.cat([G.batch[G.target_index], G.batch[layer0]], dim=0)

        return G.h[G.target_index], G.batch[G.target_index]

    def forward(self, G):
        device = self.get_device()
        G = G.to(device)

        num_nodes_batch = G.x.shape[0]
        num_layers_batch = max(G.bi_layer_index[0][0]).item() + 1
        G.h = [[torch.zeros(num_nodes_batch, self.hidden_dim).to(device)
                for _ in self.__getattr__("cells_{}".format(0))] for _ in self.dirs]
        for d in self.dirs:
            for l_idx in range(num_layers_batch):
                layer = G.bi_layer_index[d][0] == l_idx
                layer = G.bi_layer_index[d][1][layer]
                inp = G.x[layer]

                if l_idx > 0:  # no predecessors at first layer   
                    le_idx = []
                    for n in layer:
                        ne_idx = G.edge_index[1-d] == n
                        le_idx += [ne_idx.nonzero().squeeze(-1)]
                    le_idx = torch.cat(le_idx, dim=-1)
                    lp_edge_index = G.edge_index[:, le_idx]

                for i, cell in enumerate(self.__getattr__("cells_{}".format(d))):
                    # print(l_idx, i)
                    if l_idx == 0:
                        ps_h = None
                    else:
                        # D-VAE SPECIFIC
                        vids = torch.zeros(G.h[d][i].shape[0], self.num_nodes).to(device)
                        idx = torch.LongTensor(list(range(G.h[d][i].shape[0]))).to(device)
                        idx = idx.fmod(self.num_nodes).unsqueeze(-1)
                        vids.scatter_(1, idx, 1)
                        hs = torch.cat([G.h[d][i], vids], dim=-1)
                        hs1 = hs if self.agg == NA_GATED_SUM else G.h[d][i]

                        kwargs = {} if not self.agg_attn else \
                                    {"h_attn": G.x, "h_attn_q": G.x} if self.agg_attn_x else \
                                    {"h_attn": hs, "h_attn_q": torch.cat([G.h[d][i-1], vids], dim=-1) if i > 0 else G.x}  # just ignore query arg if self attn
                        node_agg = self.__getattr__("node_aggr_{}".format(d))[i]
                        ps_h = node_agg(hs1, lp_edge_index, edge_attr=None, **kwargs)[layer]
                    # print(inp.shape, ps_h.shape if ps_h is not None else "None")

                    inp = cell(inp, ps_h)
                    G.h[d][i][layer] += inp

        if not self.output_all:
            # D-VAE SPECIFIC - all have same node number
            if self.bidirectional:
                index = [i for i in range(num_nodes_batch) if i % self.num_nodes == 0]
                index1 = [i + (self.num_nodes - 1) for i in range(num_nodes_batch) if i % self.num_nodes == 0]
                h0 = torch.cat([G.h[0][l][index1] for l in range(self.num_layers)], dim=-1)
                h1 = torch.cat([G.h[1][l][index] for l in range(self.num_layers)], dim=-1)
                G.h = torch.cat([h0,h1], dim=-1)
                G.batch = G.batch[index]
                out = self.hg_unify(G.h)  # now includes layer dim reduction
            else:
                index1 = [i + (self.num_nodes - 1) for i in range(num_nodes_batch) if i % self.num_nodes == 0]
                G.h = torch.cat([G.h[0][l][index1] for l in range(self.num_layers)], dim=-1)
                G.batch = G.batch[index1]
                out = self.out_linear(G.h) if self.num_layers > 1 else G.h
        else:
            G.h = torch.cat([G.x] + [G.h[d][l] for d in self.dirs for l in range(self.num_layers)], dim=-1) if self.out_wx else \
                torch.cat([G.h[d][l] for d in self.dirs for l in range(self.num_layers)], dim=-1) if self.bidirectional else \
                    torch.cat([G.h[0][l] for l in range(self.num_layers)], dim=-1)

            if self.bidirectional:
                G.h = self.hg_unify(G.h)
            elif self.num_layers > 1:
                G.h = self.out_linear(G.h)

            out = self._readout(G.h, G.batch)

        # D-VAE SPECIFIC - return embedding
        return out

    def encode(self, G):
        if type(G) != list:
            G = [G]
        # encode graphs G into latent vectors
        b = Batch.from_data_list(G)
        Hg = self(b)

        # mu , logvar= self.fc1(Hg), self.fc2(Hg)
        y_hat = self.regression_layer(Hg)

        return y_hat.unsqueeze(2)



    def _ipropagate_to(self, G, v, propagator, H=None, reverse=False):
        assert not reverse
        # propagate messages to vertex index v for all graphs in G
        # return the new messages (states) at v
        G = [g for g in G if g.vcount() > v]
        if len(G) == 0:
            return

        if H is not None:
            idx = [i for i, g in enumerate(G) if g.vcount() > v]
            H = H[idx]

        v_types = [g.vs[v]['type'] for g in G]
        X = self._one_hot(v_types, self.nvt)
        Hv = X
        for l in range(self.num_layers):
            istr = str(l)
            H_name = 'H_forward' + istr  # name of the hidden states attribute
            H_name1 = 'H_forward' + str(l-1)
            # if h is not provided, use gated sum of v's predecessors' states as the input hidden state
            if H is None:
                H_pred = [[g.vs[x][H_name] for x in g.predecessors(v)] for g in G]
                if self.vid:
                    vids = [self._one_hot(g.predecessors(v), self.max_n) for g in G]
                # need to save basis for attention (other one has too large dim for GRU, gated sum NN can cope with that)
                if self.agg != NA_GATED_SUM:
                    H_pred1 = H_pred
                if self.vid:
                    H_pred = [[torch.cat([x[i], y[i:i + 1]], 1) for i in range(len(x))] for x, y in zip(H_pred, vids)]

                max_n_pred = max([len(x) for x in H_pred])  # maximum number of predecessors
                if max_n_pred == 0:
                    H = self._get_zero_hidden(len(G))
                else:
                    H_pred = [torch.cat(h_pred +
                                        [self._get_zeros(max_n_pred - len(h_pred), self.vs)], 0).unsqueeze(0)
                              for h_pred in H_pred]  # pad all to same length
                    H_pred = torch.cat(H_pred, 0)  # batch * max_n_pred * vs
                    H_pred1 =  H_pred if self.agg == NA_GATED_SUM else torch.cat([
                        torch.cat(h_pred + [self._get_zeros(max_n_pred - len(h_pred), self.vs-8)], 0).unsqueeze(0)
                        for h_pred in H_pred1], dim=0)

                    kwargs = {} if not self.agg_attn else \
                            {"h_attn": H_pred,
                             "h_attn_q": torch.cat([g.vs[v][H_name1] for g in G], dim=0) if l > 0 else X}  # just ignore query arg if self attn

                    node_agg = self.__getattr__("node_aggr_{}".format(0))[l]
                    H = node_agg(H_pred1, None, **kwargs)

            Hv = propagator[l](Hv, H)
            for i, g in enumerate(G):
                g.vs[v][H_name] = Hv[i:i + 1]
        return Hv


class AggConv(MessagePassing):
    def __init__(self, agg, num_relations=1, emb_dim=0, reverse=False):
        super(AggConv, self).__init__(aggr=agg, flow='target_to_source' if reverse else 'source_to_target')

        if num_relations > 1:
            assert emb_dim > 0
            self.edge_encoder = torch.nn.Linear(num_relations, emb_dim)  # assuming num_relations one hot encoded
            self.wea = True
        else:
            self.wea = False
        self.agg = agg

    def forward(self, x, edge_index, edge_attr=None, **kwargs):
        if edge_index is None:
            if self.agg == NA_MAX:
                return torch.max(x, dim=1)[0]
            elif self.agg == NA_SUM:
                return torch.sum(x, dim=1)

        edge_embedding = self.edge_encoder(edge_attr) if self.wea else None
        return self.propagate(edge_index, x=x, edge_attr=edge_embedding)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr if self.wea else x_j

    def update(self, aggr_out):
        return aggr_out


class GatedSumConv(MessagePassing):  # dvae needs outdim parameter
    def __init__(self, emb_dim, num_relations=1, reverse=False, mapper=None, gate=None):
        super(GatedSumConv, self).__init__(aggr='add', flow='target_to_source' if reverse else 'source_to_target')

        assert emb_dim > 0
        if num_relations > 1:
            self.wea = True
            self.edge_encoder = torch.nn.Linear(num_relations, emb_dim)
        else:
            self.wea = False
        self.mapper = nn.Linear(emb_dim, emb_dim) if mapper is None else mapper
        self.gate = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.Sigmoid()) if gate is None else gate

    def forward(self, x, edge_index, edge_attr=None, **kwargs):
        # HACK assume x contains only message sources
        if edge_index is None:
            h = self.gate(x) * self.mapper(x)
            return torch.sum(h, dim=1)

        edge_embedding = self.edge_encoder(edge_attr) if self.wea else None
        return self.propagate(edge_index, x=x, edge_attr=edge_embedding)

    def message(self, x_j, edge_attr):
        h_j = x_j + edge_attr if self.wea else x_j
        return self.gate(h_j) * self.mapper(h_j)

    def update(self, aggr_out):
        return aggr_out


class SelfAttnConv(MessagePassing):
    def __init__(self, emb_dim, attn_dim=0, num_relations=1, reverse=False):
        super(SelfAttnConv, self).__init__(aggr='add', flow='target_to_source' if reverse else 'source_to_target')

        assert emb_dim > 0
        attn_dim = attn_dim if attn_dim > 0 else emb_dim
        if num_relations > 1:
            self.wea = True
            self.edge_encoder = torch.nn.Linear(num_relations, attn_dim)
        else:
            self.wea = False
        self.attn_lin = nn.Linear(attn_dim, 1)

    # h_attn, edge_attr are optional
    def forward(self, h, edge_index, edge_attr=None, h_attn=None, **kwargs):
        # HACK assume x contains only message sources
        if edge_index is None:
            h_attn = h_attn if h_attn is not None else h
            attn_weights = self.attn_linear(h_attn).squeeze(-1)
            attn_weights = F.softmax(attn_weights, dim=-1)
            return torch.mm(attn_weights, h)

        edge_embedding = self.edge_encoder(edge_attr) if self.wea else None
        return self.propagate(edge_index, h=h, edge_attr=edge_embedding, h_attn=h_attn)

    def message(self, h_j, edge_attr, h_attn_j, index: Tensor, ptr: OptTensor, size_i: Optional[int]):
        h_attn = h_attn_j if h_attn_j is not None else h_j
        h_attn = h_attn + edge_attr if self.wea else h_attn
        # have to to this here instead of pre-computing a in forward because of missing edges in forward
        # we could do it in forward, but in our dags there is not much overlap in one convolution step
        # and if attn transformation linear is applied in forward we'd have to consider full X/H matrices
        # which in our case can be a lot larger
        # BUT we could move it to forward similar to pyg GAT implementation
        # ie apply two different linear to each respectively X/H, edge_attrs which yield a scalar each
        # the in message only sum those up (to obtain a single scalar) and do softmax
        a_j = self.attn_lin(h_attn)
        a_j = softmax(a_j, index, ptr, size_i)
        t = h_j * a_j
        return t

    def update(self, aggr_out):
        return aggr_out


class AttnConv(MessagePassing):
    def __init__(self, attn_q_dim, emb_dim, attn_dim=0, num_relations=1, reverse=False):
        super(AttnConv, self).__init__(aggr='add', flow='target_to_source' if reverse else 'source_to_target')

        assert attn_q_dim > 0  # for us is not necessarily equal to attn dim at first RN layer
        assert emb_dim > 0
        attn_dim = attn_dim if attn_dim > 0 else emb_dim
        if num_relations > 1:
            self.wea = True
            self.edge_encoder = torch.nn.Linear(num_relations, attn_dim)
        else:
            self.wea = False
        self.attn_lin = nn.Linear(attn_q_dim + attn_dim, 1)

    # h_attn_q is needed; h_attn, edge_attr are optional (we just use kwargs to be able to switch node aggregator above)
    def forward(self, h, edge_index, h_attn_q=None, edge_attr=None, h_attn=None, **kwargs):
        # HACK assume x contains only message sources
        if edge_index is None:
            query = torch.repeat_interleave(h_attn_q, repeats=h_attn.shape[1], dim=0)
            query = query.view(h_attn.shape[0], h_attn.shape[1], -1)
            h_attn = torch.cat((query, h_attn), -1)
            attn_weights = self.attn_lin(h_attn)
            attn_weights = attn_weights.view(h_attn_q.shape[0], -1)
            attn_weights = F.softmax(attn_weights, dim=-1)
            return torch.einsum('bi,bij->bj', attn_weights, h)

        edge_embedding = self.edge_encoder(edge_attr) if self.wea else None
        return self.propagate(edge_index, h_attn_q=h_attn_q, h=h, edge_attr=edge_embedding, h_attn=h_attn)

    def message(self, h_attn_q_i, h_j, edge_attr, h_attn_j, index: Tensor, ptr: OptTensor, size_i: Optional[int]):
        h_attn = h_attn_j if h_attn_j is not None else h_j
        h_attn = h_attn + edge_attr if self.wea else h_attn
        # see comment in above self attention why this is done here and not in forward
        a_j = self.attn_lin(torch.cat([h_attn_q_i, h_attn], dim=-1))
        a_j = softmax(a_j, index, ptr, size_i)
        t = h_j * a_j
        return t

    def update(self, aggr_out):
        return aggr_out



class DVAE(nn.Module):
    def __init__(self, max_n, nvt, START_TYPE, END_TYPE, hs=501, nz=56, bidirectional=False, vid=True):
        super(DVAE, self).__init__()
        self.max_n = max_n  # maximum number of vertices
        self.nvt = nvt  # number of vertex types
        self.START_TYPE = START_TYPE
        self.END_TYPE = END_TYPE
        self.hs = hs  # hidden state size of each vertex
        self.nz = nz  # size of latent representation z
        self.gs = hs  # size of graph state
        self.bidir = bidirectional  # whether to use bidirectional encoding
        self.vid = vid
        self.device = None
        self.n_params = 0

        if self.vid:
            self.vs = hs + max_n  # vertex state size = hidden state + vid
        else:
            self.vs = hs

        # 0. encoding-related
        self.grue_forward = nn.GRUCell(nvt, hs)  # encoder GRU
        self.grue_backward = nn.GRUCell(nvt, hs)  # backward encoder GRU
        self.fc1 = nn.Linear(self.gs, nz)  # latent mean
        # self.fc2 = nn.Linear(self.gs, nz)  # latent logvar

        # 1. decoding-related
        self.grud = nn.GRUCell(nvt, hs)  # decoder GRU
        # self.fc3 = nn.Linear(nz, hs)  # from latent z to initial hidden state h0
        self.add_vertex = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.ReLU(),
                nn.Linear(hs * 2, nvt)
                )  # which type of new vertex to add f(h0, hg)
        self.add_edge = nn.Sequential(
                nn.Linear(hs * 2, hs * 4), 
                nn.ReLU(), 
                nn.Linear(hs * 4, 1)
                )  # whether to add edge between v_i and v_new, f(hvi, hnew)

        # 2. gate-related
        self.gate_forward = nn.Sequential(
                nn.Linear(self.vs, hs), 
                nn.Sigmoid()
                )
        self.gate_backward = nn.Sequential(
                nn.Linear(self.vs, hs), 
                nn.Sigmoid()
                )
        self.mapper_forward = nn.Sequential(
                nn.Linear(self.vs, hs, bias=False),
                )  # disable bias to ensure padded zeros also mapped to zeros
        self.mapper_backward = nn.Sequential(
                nn.Linear(self.vs, hs, bias=False), 
                )

        # 3. bidir-related, to unify sizes
        if self.bidir:
            self.hv_unify = nn.Sequential(
                    nn.Linear(hs * 2, hs), 
                    )
            self.hg_unify = nn.Sequential(
                    nn.Linear(self.gs * 2, self.gs), 
                    )

        # 4. other
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.logsoftmax1 = nn.LogSoftmax(1)

    def get_device(self):
        if self.device is None:
            self.device = next(self.parameters()).device
        return self.device
    
    def _get_zeros(self, n, length):
        return torch.zeros(n, length).to(self.get_device()) # get a zero hidden state

    def _get_zero_hidden(self, n=1):
        return self._get_zeros(n, self.hs) # get a zero hidden state

    def _one_hot(self, idx, length):
        if type(idx) in [list, range]:
            if idx == []:
                return None
            idx = torch.LongTensor(idx).unsqueeze(0).t()
            x = torch.zeros((len(idx), length)).scatter_(1, idx, 1).to(self.get_device())
        else:
            idx = torch.LongTensor([idx]).unsqueeze(0)
            x = torch.zeros((1, length)).scatter_(1, idx, 1).to(self.get_device())
        return x

    def _gated(self, h, gate, mapper):
        return gate(h) * mapper(h)

    def _collate_fn(self, G):
        return [g.copy() for g in G]

    def _propagate_to(self, G, v, propagator, H=None, reverse=False):
        # propagate messages to vertex index v for all graphs in G
        # return the new messages (states) at v
        G = [g for g in G if g.vcount() > v]
        if len(G) == 0:
            return
        if H is not None:
            idx = [i for i, g in enumerate(G) if g.vcount() > v]
            H = H[idx]
        v_values = [g.vs[v]['scalar_value'] for g in G]
        X = torch.tensor(v_values, dtype=torch.float).unsqueeze(1)
        X = X.to(self.get_device())

        if reverse:
            H_name = 'H_backward'  # name of the hidden states attribute
            H_pred = [[g.vs[x][H_name] for x in g.successors(v)] for g in G]
            if self.vid:
                vids = [self._one_hot(g.successors(v), self.max_n) for g in G]
            gate, mapper = self.gate_backward, self.mapper_backward
        else:
            H_name = 'H_forward'  # name of the hidden states attribute
            H_pred = [[x for x in g.predecessors(v)] for g in G]
            # print(H_pred)
            H_pred = [[g.vs[x][H_name] for x in g.predecessors(v)] for g in G]
            if self.vid:
                vids = [self._one_hot(g.predecessors(v), self.max_n) for g in G]
            gate, mapper = self.gate_forward, self.mapper_forward
        if self.vid:
            H_pred = [[torch.cat([x[i], y[i:i+1]], 1) for i in range(len(x))] for x, y in zip(H_pred, vids)]

        if H is None:
            max_n_pred = max([len(x) for x in H_pred])  # maximum number of predecessors
            if max_n_pred == 0:
                H = self._get_zero_hidden(len(G))
            else:
                H_pred = [torch.cat(h_pred + 
                            [self._get_zeros(max_n_pred - len(h_pred), self.vs)], 0).unsqueeze(0) 
                            for h_pred in H_pred]  # pad all to same length
                H_pred = torch.cat(H_pred, 0)  # batch * max_n_pred * vs
                H = self._gated(H_pred, gate, mapper).sum(1)  # batch * hs
        Hv = propagator(X, H)
        for i, g in enumerate(G):
            g.vs[v][H_name] = Hv[i:i+1]
        
        return Hv

    def _propagate_from(self, G, v, propagator, H0=None, reverse=False):
        # perform a series of propagation_to steps starting from v following a topo order
        # assume the original vertex indices are in a topological order
        if reverse:
            prop_order = range(v, -1, -1)
        else:
            prop_order = range(v, self.max_n)
        Hv = self._propagate_to(G, v, propagator, H0, reverse=reverse)  # the initial vertex
        for v_ in prop_order[1:]:
            self._propagate_to(G, v_, propagator, reverse=reverse)
        return Hv

    def _update_v(self, G, v, H0=None):
        # perform a forward propagation step at v when decoding to update v's state
        self._propagate_to(G, v, self.grud, H0, reverse=False)
        return
    
    def _get_vertex_state(self, G, v):
        # get the vertex states at v
        Hv = []
        for g in G:
            if v >= g.vcount():
                hv = self._get_zero_hidden()
            else:
                hv = g.vs[v]['H_forward']
            Hv.append(hv)
        Hv = torch.cat(Hv, 0)
        return Hv

    def _get_graph_state(self, G, decode=False):
        # get the graph states
        # when decoding, use the last generated vertex's state as the graph state
        # when encoding, use the ending vertex state or unify the starting and ending vertex states
        Hg = []
        for g in G:
            hg = g.vs[g.vcount()-1]['H_forward']
            if self.bidir and not decode:  # decoding never uses backward propagation
                hg_b = g.vs[0]['H_backward']
                hg = torch.cat([hg, hg_b], 1)
            Hg.append(hg)
        Hg = torch.cat(Hg, 0)
        if self.bidir and not decode:
            Hg = self.hg_unify(Hg)

        return Hg

    def encode(self, G):
        # encode graphs G into latent vectors
        if type(G) != list:
            G = [G]
        self._propagate_from(G, 0, self.grue_forward, H0=self._get_zero_hidden(len(G)),
                             reverse=False)
        if self.bidir:
            self._propagate_from(G, self.max_n-1, self.grue_backward, 
                                 H0=self._get_zero_hidden(len(G)), reverse=True)
        Hg = self._get_graph_state(G)
        mu = self.fc1(Hg)
        return mu


    def _get_edge_score(self, Hvi, H, H0):
        # compute scores for edges from vi based on Hvi, H (current vertex) and H0
        # in most cases, H0 need not be explicitly included since Hvi and H contain its information
        return self.sigmoid(self.add_edge(torch.cat([Hvi, H], -1)))


    def forward(self, G):
        mu = self.encode(G)
        return mu.unsqueeze(2)

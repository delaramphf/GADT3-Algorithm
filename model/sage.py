import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

class SAGE(nn.Module):
    def __init__(self, ndim, activation, dropout, hid_dim=32):
        super(SAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(ndim, hid_dim))
        self.layers.append(SAGEConv(hid_dim, hid_dim))
        self.dropout = nn.Dropout(p=dropout)
        self.activation = activation
        self.attention_weights = [
            nn.Parameter(torch.randn(ndim, hid_dim)),
            nn.Parameter(torch.randn(hid_dim, hid_dim))
        ]
        for i, param in enumerate(self.attention_weights):
            self.register_parameter(f"attention_weight_{i}", param)
            
    def compute_attention(self, g, nfeats, adj, attention_weight):
        attention_scores = torch.matmul(nfeats, attention_weight)
        attention_scores = torch.matmul(attention_scores, attention_scores.T)
        adj_dense = adj.to_dense()
        attention_scores = attention_scores * adj_dense
        attention_scores = torch.min(attention_scores, attention_scores.T)
        attention_scores = torch.nn.functional.softmax(attention_scores, dim=-1)
        edge_index = adj.coalesce().indices()
        edge_scores = attention_scores[edge_index[0], edge_index[1]]
        attention_scores_sparse = torch.sparse_coo_tensor(
            edge_index, edge_scores, size=adj.shape, device=adj.device
        )
        return attention_scores_sparse

    def forward(self, g, nfeats, extract_features=False, extract_attentions=False):
        features = []
        attentions = []
        for i, (layer, attention_weight) in enumerate(zip(self.layers, self.attention_weights)):
            if i != 0:
                nfeats = self.dropout(nfeats)
            if 'net_upu' in g.etypes:
                etype = 'net_upu'
                adj = g.adjacency_matrix(etype=etype).coalesce().to(nfeats.device)
            else:
                adj = g.adjacency_matrix().coalesce().to(nfeats.device)
            attention_scores_sparse = self.compute_attention(g, nfeats, adj, attention_weight)
            nfeats = layer(nfeats, attention_scores_sparse)
            if extract_features:
                features.append(nfeats)
            if extract_attentions:
                attentions.append(attention_scores_sparse)
            nfeats = self.activation(nfeats)
        return nfeats, features, attentions

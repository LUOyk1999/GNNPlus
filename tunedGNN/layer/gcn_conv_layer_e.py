import torch.nn as nn
from torch_geometric.graphgym import cfg
import torch_geometric.graphgym.register as register
import torch_geometric.nn as pyg_nn
from torch_scatter import scatter
import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
import torch
from torch import Tensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    SparseTensor,
    torch_sparse,
)


class GCNConvWithEdges(GCNConv):
    def __init__(self, in_channels, out_channels, edge_dim=None, bias=True):
        super(GCNConvWithEdges, self).__init__(in_channels, out_channels, bias, add_self_loops=False, normalize=False)
        self.edge_dim = edge_dim
        # self.lin_edge = torch.nn.Linear(edge_dim, out_channels) if edge_dim is not None else None
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=False)
        
    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        # if self.lin_edge is None and x_j.size(-1) != edge_attr.size(-1):
        #     raise ValueError("Node and edge feature dimensionalities do not "
        #                      "match. Consider setting the 'edge_dim' "
        #                      "attribute of 'GCNConvWithEdges'")

        # Apply linear transformation to edge features if necessary
        # if self.lin_edge is not None:
        #     edge_attr = self.lin_edge(edge_attr)
        # print(edge_attr.shape,x_j.shape)
        # Modify the message passing to incorporate edge features (e.g., addition + ReLU)
        return (x_j + edge_attr).relu()
        # return (x_j).relu()

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None, edge_weight: OptTensor = None) -> Tensor:
        
        # Normalize the adjacency matrix if needed
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        # Apply linear transformation to node features
        x = self.lin(x)

        # Propagate the message passing with edge attributes
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, edge_weight=edge_weight, size=None)

        # Add bias if it exists
        if self.bias is not None:
            out = out + self.bias

        return out
    
class GCNConvLayer(nn.Module):
    def __init__(self, dim_in, dim_out, dropout, residual, ffn):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dropout = dropout
        self.residual = residual
        self.batch_norm = True
        self.ffn = ffn
        if self.batch_norm:
            self.bn_node_x = nn.BatchNorm1d(dim_out)
        self.act = nn.Sequential(
            register.act_dict[cfg.gnn.act](),
            nn.Dropout(self.dropout),
        )
        edge_dim = dim_in
        if edge_dim is not None:
            self.model = GCNConvWithEdges(dim_in, dim_out, edge_dim, bias=True)
        else:
            self.model = GCNConvWithEdges(dim_in, dim_out, bias=True)
        
        if self.ffn:
            # Feed Forward block.
            if self.batch_norm:
                self.norm1_local = nn.BatchNorm1d(dim_in)
            self.ff_linear1 = nn.Linear(dim_in, dim_in*2)
            self.ff_linear2 = nn.Linear(dim_in*2, dim_in)
            self.act_fn_ff = register.act_dict[cfg.gnn.act]()
            if self.batch_norm:
                self.norm2 = nn.BatchNorm1d(dim_in)
            self.ff_dropout1 = nn.Dropout(dropout)
            self.ff_dropout2 = nn.Dropout(dropout)
        
    def _ff_block(self, x):
        """Feed Forward block.
        """
        x = self.ff_dropout1(self.act_fn_ff(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def forward(self, batch):
        x_in = batch.x
        e_in = batch.edge_attr
        
        batch.x = self.model(batch.x, batch.edge_index, batch.edge_attr)
        if self.batch_norm:
            batch.x = self.bn_node_x(batch.x)
        batch.x = self.act(batch.x)
        
        if self.residual:
            batch.x = x_in + batch.x  # Residual connection.
        
        if self.ffn:
            if self.batch_norm:
                batch.x = self.norm1_local(batch.x)
            
            batch.x = batch.x + self._ff_block(batch.x)

            if self.batch_norm:
                batch.x = self.norm2(batch.x)

        return batch
    

import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.graphgym import cfg
import torch_geometric.graphgym.register as register

from torch_scatter import scatter
import torch
import torch.nn.functional as F


    
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
        
        self.model = pyg_nn.GCNConv(dim_in, dim_out, bias=True)
        
        if self.ffn:
            # Feed Forward block.
            if self.batch_norm:
                self.norm1_local = nn.BatchNorm1d(dim_out)
            self.ff_linear1 = nn.Linear(dim_out, dim_out*2)
            self.ff_linear2 = nn.Linear(dim_out*2, dim_out)
            self.act_fn_ff = register.act_dict[cfg.gnn.act]()
            if self.batch_norm:
                self.norm2 = nn.BatchNorm1d(dim_out)
            self.ff_dropout1 = nn.Dropout(dropout)
            self.ff_dropout2 = nn.Dropout(dropout)
        
    def _ff_block(self, x):
        """Feed Forward block.
        """
        x = self.ff_dropout1(self.act_fn_ff(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def forward(self, batch):
        x_in = batch.x
        
        batch.x = self.model(batch.x, batch.edge_index)
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
    

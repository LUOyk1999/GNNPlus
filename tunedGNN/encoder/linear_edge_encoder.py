import torch
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_edge_encoder


@register_edge_encoder('LinearEdge')
class LinearEdgeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        if cfg.dataset.name in ['MNIST', 'CIFAR10']:
            if cfg.dataset.node_encoder_name == 'LinearNode':
                self.in_dim = 1
            else:
                self.in_dim = list(eval(cfg.posenc_RWSE.kernel.times_func))[-1] + 1
        else:
            self.in_dim = list(eval(cfg.posenc_RWSE.kernel.times_func))[-1]
        # else:
        #     raise ValueError("Input edge feature dim is required to be hardset "
        #                      "or refactored to use a cfg option.")
        # print(self.in_dim)
        self.encoder = torch.nn.Linear(self.in_dim, emb_dim)

    def forward(self, batch):
        # print(batch.edge_attr.shape,batch.edge_attr)
        batch.edge_attr = self.encoder(batch.edge_attr.view(-1, self.in_dim))
        return batch

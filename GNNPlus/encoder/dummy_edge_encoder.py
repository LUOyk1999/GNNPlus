import torch
from torch_geometric.graphgym.register import register_edge_encoder
from torch_geometric.graphgym import cfg

@register_edge_encoder('DummyEdge')
class DummyEdgeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        self.encoder = torch.nn.Embedding(num_embeddings=1,
                                          embedding_dim=emb_dim)
        # torch.nn.init.xavier_uniform_(self.encoder.weight.data)
        if cfg.dataset.node_encoder_name == 'LinearNode+RWSE':
            self.in_dim = list(eval(cfg.posenc_RWSE.kernel.times_func))[-1]
            # print(self.in_dim)
            self.encoder2 = torch.nn.Linear(self.in_dim, emb_dim)

    def forward(self, batch):
        # print(batch.edge_attr.shape)
        dummy_attr = batch.edge_index.new_zeros(batch.edge_index.shape[1])
        if cfg.dataset.node_encoder_name == 'LinearNode+RWSE':
            batch.edge_attr = self.encoder(dummy_attr) + self.encoder2(batch.edge_attr.view(-1, self.in_dim))
        else:
            batch.edge_attr = self.encoder(dummy_attr)
        return batch

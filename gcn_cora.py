# gcn_cora.py
import torch, torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GCNConv

ds = Planetoid(root="data/Planetoid", name="Cora", transform=NormalizeFeatures())
data = ds[0].to('cpu')

class GCN(torch.nn.Module):
    def __init__(self, F_in, F_hid, F_out):
        super().__init__()
        self.c1 = GCNConv(F_in, F_hid)
        self.c2 = GCNConv(F_hid, F_out)
    def forward(self, x, ei):
        x = self.c1(x, ei).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        return self.c2(x, ei)

m = GCN(ds.num_features, 16, ds.num_classes)
opt = torch.optim.Adam(m.parameters(), lr=0.01, weight_decay=5e-4)

for epoch in range(1, 201):
    m.train(); opt.zero_grad()
    out = m(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask]); loss.backward(); opt.step()
    if epoch % 20 == 0:
        m.eval(); pred = out.argmax(dim=1)
        acc = (pred[data.test_mask]==data.y[data.test_mask]).float().mean().item()
        print(f"Epoch {epoch:03d} | loss {loss:.3f} | test {acc:.3f}")

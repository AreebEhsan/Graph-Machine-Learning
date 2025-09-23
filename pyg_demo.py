# quick_pyg_demo.py
import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

# 1) Build a tiny graph in NetworkX
G = nx.karate_club_graph()  # 34 nodes, undirected
nodes = list(G.nodes())
idx = {n: i for i, n in enumerate(nodes)}

# 2) edge_index (shape [2, E], dtype long)
edges = [(idx[u], idx[v]) for u, v in G.edges()]
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

# 3) Node features x (here: degree as a single feature)
x = torch.tensor([[G.degree(n)] for n in nodes], dtype=torch.float)

# 4) Make a PyG Data object
data = Data(x=x, edge_index=edge_index)

# 5) A tiny GraphSAGE and a forward pass
class TinySAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden=16, out_channels=8):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, out_channels)

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

model = TinySAGE(in_channels=data.x.size(1))
with torch.no_grad():
    z = model(data.x, data.edge_index)  # node embeddings [34, 8]
print("Embeddings shape:", z.shape)

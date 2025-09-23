# quick_pyg_demo.py
import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

#  Building a tiny graph in NetworkX
G = nx.karate_club_graph()  # 34 nodes, undirected
nodes = list(G.nodes())
idx = {n: i for i, n in enumerate(nodes)}

# edge_index (shape [2, E], dtype long)
edges = [(idx[u], idx[v]) for u, v in G.edges()]
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

#  Node features x (here: degree as a single feature)
x = torch.tensor([[G.degree(n)] for n in nodes], dtype=torch.float)

# Make a PyG Data object
data = Data(x=x, edge_index=edge_index)

# A tiny GraphSAGE and a forward pass
class TinySAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden=16, out_channels=8): # 2-layer GraphSAGE
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden) # First layer
        self.conv2 = SAGEConv(hidden, out_channels) # Second layer

    def forward(self, x, edge_index): # x: node features, edge_index: graph connectivity
        x = torch.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

model = TinySAGE(in_channels=data.x.size(1))
with torch.no_grad():
    z = model(data.x, data.edge_index) 
print("Embeddings shape:", z.shape)


 # node embeddings are [34, 8]
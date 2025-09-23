
import torch
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import SAGEConv
from dataset import MyOwnDataset

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden=64, out_channels=64, num_layers=2):
        super().__init__() # 2-layer GraphSAGE by default
        self.convs = torch.nn.ModuleList() # List of convolutional layers
        self.convs.append(SAGEConv(in_channels, hidden))
        for _ in range(num_layers-2): # hidden layers
            self.convs.append(SAGEConv(hidden, hidden))
        self.convs.append(SAGEConv(hidden, out_channels)) # final layer

    def forward(self, x, edge_index): # x: node features, edge_index: graph connectivity
        for conv in self.convs[:-1]: # all but last layer
            x = torch.relu(conv(x, edge_index)) # ReLU activation
        return self.convs[-1](x, edge_index)

# Load preprocessed data
ds = MyOwnDataset(root="data/my_graph")     # 
data = ds[0]
 
 # For link prediction, we need positive and negative edges.
edge_label_index = data.edge_index  # example: predict existing links
edge_label = torch.ones(edge_label_index.size(1))  # all positives for demo

# Add negative edges (non-existing links)
loader = LinkNeighborLoader(
    data,
    batch_size=1024,
    num_neighbors=[10, 10],
    edge_label_index=edge_label_index,
    edge_label=edge_label,
    neg_sampling_ratio=0.5
)
 # Define model and optimizer
model = GraphSAGE(in_channels=data.x.size(1)).to('cpu')
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for batch in loader:
    opt.zero_grad()
    h = model(batch.x, batch.edge_index)
    # Retrieve embeddings for the two endpoints of each (positive+negative) edge:
    src, dst = batch.edge_label_index
    score = (h[src] * h[dst]).sum(dim=-1)  # dot product scorer (toy)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(score, batch.edge_label.float())
    loss.backward()
    opt.step()

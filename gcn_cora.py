# gcn_cora.py

# cora: citation network dataset: academic papers as nodes and citationlinks as edges
import torch, torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GCNConv

ds = Planetoid(root="data/Planetoid", name="Cora", transform=NormalizeFeatures()) # Loading Cora dataset from the Planetoid collection
data = ds[0].to('cpu') # [0] -> gets first graph


# Defining a custom 2 layer GCN model
class GCN(torch.nn.Module):
    def __init__(self, F_in, F_hid, F_out): # F_in: input feature dimension, F_hid: hidden layer dimension, F_out: output feature dimension
        super().__init__()
        self.c1 = GCNConv(F_in, F_hid) #First GCN layer
        self.c2 = GCNConv(F_hid, F_out) # Second GCN layer


    def forward(self, x, ei): # x: node features, ei: edge index
        x = self.c1(x, ei).relu() # First GCN layer with ReLU activation
        x = F.dropout(x, p=0.5, training=self.training)
        return self.c2(x, ei)

m = GCN(ds.num_features, 16, ds.num_classes) # Model instance: input features, hidden layer size, number of classes
opt = torch.optim.Adam(m.parameters(), lr=0.01, weight_decay=5e-4) # Adam optimizer, weight decay regulariation technique to prevent overfitting


# Training loop to 200 epochs
for epoch in range(1, 201): #
    m.train(); opt.zero_grad() # Set model to training mode and zero gradients
    out = m(data.x, data.edge_index) # Forward pass
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask]); loss.backward(); opt.step() # Compute loss, backpropagate, and update weights
    if epoch % 20 == 0: # Evaluating every 20 epochs on test set
        m.eval(); pred = out.argmax(dim=1)  # Set model to evaluation mode and get predictions
        acc = (pred[data.test_mask]==data.y[data.test_mask]).float().mean().item() # Calculate accuracy on test set
        print(f"Epoch {epoch:03d} | loss {loss:.3f} | test {acc:.3f}")

# dataset.py
import torch
from torch_geometric.data import InMemoryDataset, Data
import networkx as nx

class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        # 1) Load your graph (example: from edgelist / json)
        G = nx.read_gpickle(self.raw_paths[0])  # or however you store it

        # 2) Map nodes to 0..N-1
        nodes = list(G.nodes())
        idx = {n: i for i, n in enumerate(nodes)}

        # 3) Build edge_index [2, E]
        import numpy as np
        edges = np.array([(idx[u], idx[v]) for u, v in G.edges()], dtype=np.int64)
        edge_index = torch.tensor(edges.T, dtype=torch.long)  # shape [2, E]

        # 4) Build x (toy: degree as a single feature)
        deg = torch.tensor([G.degree(n) for n in nodes], dtype=torch.float).unsqueeze(1)  # [N,1]

        # 5) Optional labels (e.g., node labels) or leave out
        data = Data(x=deg, edge_index=edge_index)

        torch.save(self.collate([data]), self.processed_paths[0])

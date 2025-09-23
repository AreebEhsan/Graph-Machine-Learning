# dataset.py
import torch
from torch_geometric.data import InMemoryDataset, Data
import networkx as nx

# Custom dataset class for loading a graph from a NetworkX object;  efficiently load and store a single graph as standard format for GNNs.
class MyOwnDataset(InMemoryDataset):
    # This is the constructor for the dataset.
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        # `root`: The directory where the processed dataset will be saved.
        super().__init__(root, transform, pre_transform, pre_filter)
        # This line is crucial for an InMemoryDataset. It checks for a processed file at self.processed_paths[0].
        # If the file exists, it loads the data and slices; otherwise, it triggers the `process()` method.
        self.data, self.slices = torch.load(self.processed_paths[0])

    # This property returns a list of file names that will be generated after processing.
    # PyG checks for the existence of these files to determine if processing is needed.
    @property
    def processed_file_names(self):
        return ['data.pt']

    # This method contains the logic for converting the raw data into the PyTorch Geometric format.
    # It's automatically called by the constructor if the processed data file ('data.pt') is not found.
    def process(self):
        # 1) Load your graph (example: from edgelist / json)
        # This line loads a graph from a serialized NetworkX object file.
        # `self.raw_paths[0]` points to the raw data file in the `raw` subdirectory.
        G = nx.read_gpickle(self.raw_paths[0])  # or however you store it

        # 2) Map nodes to 0..N-1
        # PyTorch Geometric requires node indices to be contiguous, starting from 0.
        nodes = list(G.nodes())
        idx = {n: i for i, n in enumerate(nodes)}

        # 3) Build edge_index [2, E]
        # `edge_index` is the most important component, defining the graph's connectivity.
        import numpy as np
        # This list comprehension iterates through all edges in the NetworkX graph and maps their original
        # node names to the new 0-based integer indices.
        edges = np.array([(idx[u], idx[v]) for u, v in G.edges()], dtype=np.int64)
        # The edge list is then transposed to the required [2, num_edges] format and converted to a PyTorch tensor.
        edge_index = torch.tensor(edges.T, dtype=torch.long)  # shape [2, E]

        # 4) Build x (toy: degree as a single feature)
        # `x` is the node feature matrix with shape [num_nodes, num_features].
        # Here, a simple example is used: the feature of each node is its degree.
        deg = torch.tensor([G.degree(n) for n in nodes], dtype=torch.float).unsqueeze(1)  # [N,1]
        # `.unsqueeze(1)` is used to add a dimension, changing the shape from [N] to [N, 1],
        # which is required for a feature matrix with a single feature.


        # This line assembles all the necessary tensors into a single `Data` object.
        data = Data(x=deg, edge_index=edge_index)
        # If node labels (`y`) or edge features (`edge_attr`) were available, they could be passed here.

        # `collate` combines the single `Data` object into a list.
        # `torch.save` serializes the data and saves it to the processed file, `data.pt`.
        torch.save(self.collate([data]), self.processed_paths[0])
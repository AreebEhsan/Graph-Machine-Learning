# 🧠 Understanding Graph Neural Networks (GNNs)

This repository contains summarized notes and practical insights into **Graph Neural Networks (GNNs)** — covering both the **theoretical foundation** and **hands-on implementation** using **PyTorch Geometric (PyG)**.  
It is divided into two main articles followed by concise **short notes** and **practical implementation**.

---

## 📘  1: The Theory Behind Graph Neural Networks

### What is a Graph?
A **graph** in GNNs is a **non-Euclidean data structure** made up of:
- **Nodes (vertices):** Represent entities  
- **Edges (links):** Represent relationships between entities  

Unlike regular grids (e.g., images), graphs are **irregular** and **non-uniform**, allowing GNNs to model real-world relationships such as molecules, users, or networks.

---

### 🧩 The GNN Family Tree

#### **1. Graph Convolutional Networks (GCNs)**
- Foundation of all GNNs.  
- Use **convolution** to aggregate information from neighboring nodes.  
- **Transductive** → Best for static graphs (don’t generalize well to new nodes).

**Two main types:**
- **Spectral Convolution:**  
  Transforms the graph using the **Laplacian Matrix** and **Fourier Transform**.  
  Good for analyzing the full graph but computationally heavy.
- **Spatial Convolution:**  
  Aggregates directly from local neighbors.  
  More intuitive and efficient for local structures.

#### **2. GraphSAGE (Graph Sample and Aggregate)**
- Designed to overcome GCN’s transductive limitation.  
- **Inductive:** Learns general functions for unseen nodes.  
- Samples and aggregates fixed-size neighborhoods → great for dynamic graphs.

#### **3. Graph Attention Networks (GATs)**
- Introduce **attention mechanisms** to weigh neighbors differently.  
- Mimics how humans focus on important connections.  

**Types of attention:**
- **Global Attention:** Every node attends to every other node (expensive).  
- **Mask Attention:** Only attends to direct neighbors (efficient).

---

### 🌍 Real-World Applications of GNNs
- **Biochemistry:** Protein interaction and drug discovery.  
- **Recommendation Systems:** Used by Pinterest, JD.com, etc.  
- **Skeletal Motion Recognition:** ST-GNNs capture spatial + temporal data.

---

## 💻  2: Practical Implementation with PyTorch-Geometric

### Step 1: Creating a Custom Dataset
GNNs in PyG use the **`Data` object** as their main container:
- **`x`** → Node feature matrix `[num_nodes, num_features]`  
- **`edge_index`** → List of edges `[2, num_edges]`  
  - Each column = (source, target) node indices  

## 🧾 Short Notes

| Concept | Explanation |
|----------|--------------|
| **Graph** | Nodes (entities) + Edges (relationships) |
| **Node Features (`x`)** | `[num_nodes, num_features]` numeric attributes per node |
| **Edge Index (`edge_index`)** | `[2, num_edges]` — lists connections like `[[0,1],[1,2]]` |
| **Labels (`y`)** | Target prediction (node class, edge existence, etc.) |
| **Message Passing** | Node updates by aggregating neighbor info |
| **Embedding** | Learned vector for each node |
| **GCN** | Averages neighbor features (fast, simple) |
| **GraphSAGE** | Samples fixed neighbors → inductive learning |
| **GAT** | Learns attention weights for neighbors |
| **Homophily** | Connected nodes tend to be similar |
| **Heterophily** | Connected nodes differ (challenging) |
| **Over-smoothing** | Too many layers → all nodes similar |
| **Over-squashing** | Long-range info bottlenecked |
| **Transductive** | Test nodes already in graph (Cora) |
| **Inductive** | Test nodes unseen (GraphSAGE) |
| **Mini-batch / Neighbor Sampling** | Train using subgraphs for efficiency |
| **Negative Sampling** | Add fake edges for link prediction |
| **Masks** | Boolean arrays for train/val/test splits |

---

### 🧠 Visualizing Message Passing (1 Layer)
For each node *i*:
1. Gather neighbor vectors (and optionally itself).  
2. Aggregate (mean/sum/attention).  
3. Apply linear layer + activation → new representation.

---

###  Minimal PyG Setup
```python
from torch_geometric.data import Data

# Node features: [N, F], edges: [2, E]
data = Data(x=x, edge_index=edge_index, y=labels)
```

---

###  Three Major Models at a Glance
| Model | Summary | Key Strength |
|--------|----------|---------------|
| **GCN** | Average neighbors → linear layer | Simple baseline |
| **GraphSAGE** | Sample k neighbors per hop | Scalable, inductive |
| **GAT** | Learn neighbor importance (attention) | High accuracy, costlier |

---

### 🎯 Common GNN Tasks
- **Node Classification** → Label each node  
- **Link Prediction** → Predict if edge exists (u,v)  
- **Graph Classification** → Label entire graphs (e.g., molecules)

---

## 🧠 Aggregation, Scaling, and Storage

### 🔄 Aggregation
Combines neighbor feature vectors using:
- **Mean:** captures average neighborhood traits  
- **Sum:** preserves degree importance  
- **Max-pooling:** focuses on dominant features  

### ⚖️ Scaling
Preprocessing step to normalize numerical features to a common range.  
Ensures no single feature dominates (e.g., income vs age).

Common methods:  
- **Normalization**
- **Standardization**

### 💾 Efficient Graph Storage in PyG
PyG stores graphs in **COO (Coordinate)** format using:
- **`edge_index`** → Source & target node pairs  
- **`edge_attr`** → Optional edge features  

This format is memory-efficient for large, sparse graphs compared to adjacency matrices.



### 🧭 Summary
> **Graph Neural Networks** enable deep learning on relational data.  
> They combine topology + features to learn powerful representations for complex domains — from molecules to social networks.  
> Libraries like **PyTorch Geometric** make implementing GNNs accessible, scalable, and efficient.

---

### 🧑‍💻 Author
**Areeb Ehsan**  
> *Computer Science @ Georgia State University*  
> 📍 [areebehsan.dev](https://areebehsan.dev)


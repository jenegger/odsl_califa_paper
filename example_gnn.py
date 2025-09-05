import torch
from torch_geometric.data import Data, InMemoryDataset, DataLoader
from torch_geometric.utils import dense_to_sparse
import numpy as np
#1) Dataset
class HitsDataset(InMemoryDataset):
    def __init__(self, events, transform=None):
        """
        events: list of (X, cluster_ids) pairs
            X: np.array shape (N, 3) = features
            cluster_ids: np.array shape (N,) = cluster membership per hit
        """
        self.events = events
        super().__init__('.', transform, None, None)
        self.data, self.slices = self.collate([self._make_graph(x, c) for x, c in events])

    def _make_graph(self, X, cluster_ids):
        N = X.shape[0]
        x = torch.tensor(X, dtype=torch.float)

        # fully connected graph
        adj = torch.ones((N, N)) - torch.eye(N)
        edge_index, _ = dense_to_sparse(adj)

        # edge labels: 1 if same cluster, else 0
        ci = torch.tensor(cluster_ids)
        edge_label = (ci[edge_index[0]] == ci[edge_index[1]]).float()

        return Data(x=x, edge_index=edge_index, edge_label=edge_label)

#2)GNN + Edge Predictor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class HitGNN(nn.Module):
    def __init__(self, in_dim=3, hidden_dim=32, emb_dim=16):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, emb_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x  # embeddings per hit

class EdgePredictor(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2*emb_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, z, edge_index):
        z_i = z[edge_index[0]]
        z_j = z[edge_index[1]]
        z_cat = torch.cat([z_i, z_j], dim=-1)
        return torch.sigmoid(self.fc(z_cat)).squeeze()


#3) Training Loop
def train(model, predictor, loader, optimizer, device):
    model.train()
    predictor.train()
    total_loss = 0
    criterion = nn.BCELoss()

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        z = model(data)
        pred = predictor(z, data.edge_index)
        loss = criterion(pred, data.edge_label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def test(model, predictor, loader, device):
    model.eval()
    predictor.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            z = model(data)
            pred = predictor(z, data.edge_index)
            correct += ((pred > 0.5) == data.edge_label.bool()).sum().item()
            total += data.edge_label.size(0)
    return correct / total

#4) Example Run
# fake dataset: 100 events, each with 6–12 hits, 2–3 clusters
events = []
for _ in range(100):
    n_hits = np.random.randint(6, 12)
    n_clusters = np.random.randint(2, 4)
    cluster_ids = np.random.randint(0, n_clusters, size=n_hits)
    X = np.random.rand(n_hits, 3)  # (energy, theta, phi)
    events.append((X, cluster_ids))

dataset = HitsDataset(events)
train_loader = DataLoader(dataset[:80], batch_size=1, shuffle=True)
test_loader  = DataLoader(dataset[80:], batch_size=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HitGNN(in_dim=3, hidden_dim=32, emb_dim=16).to(device)
predictor = EdgePredictor(emb_dim=16).to(device)
optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=0.01)

for epoch in range(1, 21):
    loss = train(model, predictor, train_loader, optimizer, device)
    acc  = test(model, predictor, test_loader, device)
    print(f"Epoch {epoch:02d}, Loss: {loss:.4f}, Test Acc: {acc:.4f}")
#5) Inverence = Build Clusters
from torch_geometric.utils import to_networkx
import networkx as nx

def get_clusters(model, predictor, data, threshold=0.5):
    model.eval()
    predictor.eval()
    with torch.no_grad():
        z = model(data)
        probs = predictor(z, data.edge_index)
    # keep edges above threshold
    mask = probs > threshold
    edge_index = data.edge_index[:, mask]
    # turn into networkx graph and get connected components
    G = to_networkx(Data(edge_index=edge_index, num_nodes=data.x.size(0)))
    clusters = list(nx.connected_components(G))
    return clusters



#6) how to use the get clusters stuff on a single event
# assume model + predictor are trained and on same device
event = dataset[0]        # pick first event
event = event.to(device)  # move to GPU/CPU

clusters = get_clusters(model, predictor, event, threshold=0.5)

print("Predicted clusters for event 0:")
for i, c in enumerate(clusters):
    print(f"Cluster {i}: hits {sorted(list(c))}")

#7) and running ove all test events...
for idx, data in enumerate(test_loader):
    data = data.to(device)
    clusters = get_clusters(model, predictor, data, threshold=0.5)
    print(f"\nEvent {idx}:")
    for i, c in enumerate(clusters):
        print(f"  Cluster {i}: hits {sorted(list(c))}")


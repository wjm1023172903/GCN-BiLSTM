from torch_geometric.loader import DataLoader
from EarlyStopping import EarlyStopping
from dataset import GraphTimeDataset
from GCN_bilstm import GCNBiLSTM

dataset = GraphTimeDataset('totc.xlsx')
train_loader = DataLoader(dataset[:200], batch_size=8, shuffle=True)
val_loader = DataLoader(dataset[200:], batch_size=8)

model = GCNBiLSTM(
    num_node_features=dataset[0].num_node_features,
    gcn_hidden=16,
    lstm_hidden=32,
    output_dim=1
)

sample_data = dataset[0]
model.set_graph_matrices(sample_data.adj_matrix, sample_data.degree_matrix)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
early_stopping = EarlyStopping(patience=15, verbose=True)

for epoch in range(1500):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            pred = model(batch.x, batch.edge_index)
            val_loss += criterion(pred, batch.y).item()

    early_stopping(val_loss / len(val_loader), model)
    if early_stopping.early_stop:
        print("Early stopping triggered")
        break

model.load_state_dict(torch.load('checkpoint.pt'))
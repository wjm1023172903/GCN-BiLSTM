import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from dataset import load_gcn_data, generate_graph_data, load_bilstm_data
from early_stopping import EarlyStopping

tz_tensor, mb_tensor = load_gcn_data('js.csv')
graph_data = generate_graph_data(tz_tensor, mb_tensor)

trainX, trainY, testX, testY, scaler = load_bilstm_data('totc.csv')

gcn_params = {
    "num_node_features": tz_tensor.shape[1],
    "gcn_hidden_channels": 16,
    "output_dim": 16
}

bilstm_params = {
    "input_size": trainX.shape[2],
    "hidden_layer_size": 64,
    "output_size": 16
}

final_output_dim = 1

model = GCNBiLSTM(gcn_params, bilstm_params, final_output_dim)
model.set_graph_matrices(graph_data.adj_matrix, graph_data.degree_matrix)

graph_dataloader = DataLoader([graph_data], batch_size=1)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
criterion = nn.MSELoss()

early_stopper = EarlyStopping(patience=50)

for epoch in range(10000):
    model.train()
    total_loss = 0
    for batch in graph_dataloader:
        optimizer.zero_grad()

        graph_data = batch
        sequence_data = trainX
        out = model((graph_data.x, graph_data.edge_index, graph_data.batch), sequence_data)

        loss = criterion(out, trainY.view(-1, final_output_dim))
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(graph_dataloader)
    print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

    if early_stopper.step(avg_loss):
        print("Early stopping triggered.")
        break
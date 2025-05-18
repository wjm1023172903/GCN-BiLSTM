
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from GCN_bilstm import GCN, BiLSTMModel
from dataset import load_gcn_data, generate_graph_data, load_bilstm_data
from early_stopping import EarlyStopping
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import math

# ===== GCN TRAINING =====
tz_tensor, mb_tensor = load_gcn_data('2001.xlsx')
graph_data = generate_graph_data(tz_tensor, mb_tensor)
model = GCN(tz_tensor.shape[1], 16, 1)
model.set_graph_matrices(graph_data.adj_matrix, graph_data.degree_matrix)
dataloader = DataLoader([graph_data], batch_size=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()
early_stopper = EarlyStopping(patience=50)

for epoch in range(10000):
    for batch in dataloader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    if early_stopper.step(loss.item()):
        print("Early stopping triggered.")
        break

# ===== BiLSTM TRAINING =====
trainX, trainY, testX, testY, scaler = load_bilstm_data('totc.xlsx')
bilstm = BiLSTMModel(trainX.shape[2], 64, 1)
optimizer = optim.Adam(bilstm.parameters(), lr=0.0009, weight_decay=1e-5)
criterion = nn.MSELoss()
early_stopper_bilstm = EarlyStopping(patience=50)

for epoch in range(1200):
    bilstm.train()
    bilstm.hidden_cell = (
        torch.zeros(2, trainX.size(0), bilstm.hidden_layer_size),
        torch.zeros(2, trainX.size(0), bilstm.hidden_layer_size)
    )
    y_pred = bilstm(trainX)
    loss = criterion(y_pred, trainY.view(-1, 1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    if early_stopper_bilstm.step(loss.item()):
        print("Early stopping BiLSTM.")
        break

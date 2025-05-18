import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# Graph Convolutional Network
class GCN(nn.Module):
    def __init__(self, num_node_features, gcn_hidden_channels, output_dim):
        super(GCN, self).__init__()
        self.gcn1 = GCNConv(num_node_features, gcn_hidden_channels)
        self.gcn2 = GCNConv(gcn_hidden_channels, gcn_hidden_channels)
        self.fc = nn.Linear(gcn_hidden_channels, output_dim)
        self.adj_matrix = None

    def set_graph_matrices(self, adj_matrix, degree_matrix):
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(degree_matrix))
        self.adj_matrix = torch.matmul(torch.matmul(D_inv_sqrt, adj_matrix), D_inv_sqrt)

    def forward(self, x, edge_index, batch):
        x = self.gcn1(x, edge_index)
        x = F.relu(x)
        x = self.gcn2(x, edge_index)
        x = F.relu(x)
        if self.adj_matrix is not None:
            x = torch.matmul(self.adj_matrix, x)
        out = self.fc(x)
        return out

# BiLSTM for sequence modeling
class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size=256, output_size=1):
        super(BiLSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.bilstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(hidden_layer_size * 2, output_size)
        self.hidden_cell = None

    def forward(self, input_seq):
        bilstm_out, self.hidden_cell = self.bilstm(input_seq, self.hidden_cell)
        bilstm_out = self.dropout(bilstm_out)
        predictions = self.linear(bilstm_out[:, -1])
        return predictions

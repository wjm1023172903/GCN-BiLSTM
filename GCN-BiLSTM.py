import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GCNBiLSTM(nn.Module):
    def __init__(self, gcn_params, bilstm_params, final_output_dim):
        super(GCNBiLSTM, self).__init__()

        self.gcn = GCN(
            num_node_features=gcn_params["num_node_features"],
            gcn_hidden_channels=gcn_params["gcn_hidden_channels"],
            output_dim=gcn_params["output_dim"]
        )

        self.bilstm = BiLSTMModel(
            input_size=bilstm_params["input_size"],
            hidden_layer_size=bilstm_params["hidden_layer_size"],
            output_size=bilstm_params["output_size"]
        )

        combined_dim = gcn_params["output_dim"] + bilstm_params["output_size"]
        self.fc = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Linear(128, final_output_dim)
        )

    def set_graph_matrices(self, adj_matrix, degree_matrix):
        self.gcn.set_graph_matrices(adj_matrix, degree_matrix)

    def forward(self, graph_data, sequence_data):
        x, edge_index, batch = graph_data
        gcn_out = self.gcn(x, edge_index, batch)  # [num_nodes, gcn_output_dim]
        graph_embedding = global_mean_pool(gcn_out, batch)  # [batch_size, gcn_output_dim]
        bilstm_out = self.bilstm(sequence_data)  # [batch_size, lstm_output_size]
        combined = torch.cat([graph_embedding, bilstm_out], dim=1)  # [batch_size, combined_dim]
        out = self.fc(combined)
        return out


# 保持原始GCN和BiLSTM实现不变
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


class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size=256, output_size=1):
        super(BiLSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.bilstm = nn.LSTM(input_size, hidden_layer_size,
                              batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(hidden_layer_size * 2, output_size)

    def forward(self, input_seq):
        bilstm_out, _ = self.bilstm(input_seq)
        bilstm_out = self.dropout(bilstm_out)
        predictions = self.linear(bilstm_out[:, -1])
        return predictions
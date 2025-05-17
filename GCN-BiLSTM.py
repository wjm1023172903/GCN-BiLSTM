import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNBiLSTM(nn.Module):
    def __init__(self, num_node_features, gcn_hidden, lstm_hidden, output_dim):
        super().__init__()
        self.gcn1 = GCNConv(num_node_features, gcn_hidden)
        self.gcn2 = GCNConv(gcn_hidden, gcn_hidden)

        self.bilstm = nn.LSTM(gcn_hidden, lstm_hidden,
                              bidirectional=True,
                              batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

        self.adj_matrix = None

    def set_graph_matrices(self, adj_matrix, degree_matrix):
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(degree_matrix))
        self.adj_matrix = torch.matmul(torch.matmul(D_inv_sqrt, adj_matrix), D_inv_sqrt)

    def forward(self, x, edge_index, seq_length=1):

        x = F.relu(self.gcn1(x, edge_index))
        x = F.relu(self.gcn2(x, edge_index))

        if self.adj_matrix is not None:
            x = torch.matmul(self.adj_matrix, x)

        x = x.unsqueeze(1).repeat(1, seq_length, 1)


        bilstm_out, _ = self.bilstm(x)
        output = self.fc(bilstm_out[:, -1, :])
        return output
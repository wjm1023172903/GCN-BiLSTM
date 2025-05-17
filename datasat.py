import torch
from torch_geometric.data import Data, Dataset


def generate_graph_data(tz_tensor, mb_tensor):
    num_nodes = 13
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 9, 9, 10],
                               [3, 9, 10, 3, 4, 11, 6, 11, 12, 10, 5, 8, 11, 7, 8, 12, 8, 9, 12, 9, 11, 10, 11, 11]],
                              dtype=torch.long)

    adj_matrix = torch.tensor([
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0],
        [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0],
        [1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
    ], dtype=torch.float)

    degree_matrix = torch.tensor([3, 4, 2, 4, 3, 3, 4, 4, 5, 5, 5, 5, 3], dtype=torch.float)

    return Data(
        x=tz_tensor[:num_nodes],
        edge_index=edge_index,
        y=mb_tensor[:num_nodes],
        adj_matrix=adj_matrix,
        degree_matrix=degree_matrix
    )

    def __len__(self):
        return len(self.mb_data) - self.look_back - 1

    def __getitem__(self, idx):
        tz_slice = self.tz_data[idx:idx + self.look_back]
        mb_target = self.mb_data[idx + self.look_back]

        graph = Data(
            x=tz_slice[-1],
            edge_index=self.edge_index,
            y=mb_target
        )
        return graph
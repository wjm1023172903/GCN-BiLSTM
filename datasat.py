import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from sklearn.preprocessing import MinMaxScaler

def load_gcn_data(path):
    data = pd.read_excel(path, header=None)
    mb_data = data.iloc[:, 0]
    tz_data = data.iloc[:, 1:]
    mb_tensor = torch.tensor(mb_data.values, dtype=torch.float32).unsqueeze(1)
    tz_tensor = torch.tensor(tz_data.values, dtype=torch.float32)
    return tz_tensor, mb_tensor

def generate_graph_data(tz_tensor, mb_tensor):
    num_nodes = 13
    x = tz_tensor[:num_nodes]
    y = mb_tensor[:num_nodes]
    edge_index = torch.tensor([
        [0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 9, 9, 10],
        [3, 9, 10, 3, 4, 11, 6, 11, 12, 10, 5, 8, 11, 7, 8, 12, 8, 9, 12, 9, 11, 10, 11, 11]
    ], dtype=torch.long)

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
    data = Data(x=x, edge_index=edge_index, y=y, adj_matrix=adj_matrix, degree_matrix=degree_matrix)
    return data

def load_bilstm_data(path, look_back=1):
    df = pd.read_excel(path)
    dataset = df.values.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    train_size = int(len(dataset) * 0.8)
    train, test = dataset[:train_size], dataset[train_size:]

    def create_dataset(dataset):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - 1):
            dataX.append(dataset[i:(i + look_back), 3:])
            dataY.append(dataset[i + look_back, 2])
        return np.array(dataX), np.array(dataY)

    trainX, trainY = create_dataset(train)
    testX, testY = create_dataset(test)

    return (
        torch.from_numpy(trainX).float(),
        torch.from_numpy(trainY).float(),
        torch.from_numpy(testX).float(),
        torch.from_numpy(testY).float(),
        scaler
    )

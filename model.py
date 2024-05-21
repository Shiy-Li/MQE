import torch.nn as nn


class MQENet(nn.Module):
    def __init__(self, z_dim, feat_dim, hid_dim):
        super(MQENet, self).__init__()
        # Expectation
        self.E_MLP = nn.Sequential(nn.Linear(z_dim, hid_dim[0]), nn.ReLU(),
                                   nn.Linear(hid_dim[0], feat_dim))
        # Variance (Quality)
        self.Q_MLP = nn.Sequential(nn.Linear(z_dim, hid_dim[1]), nn.ReLU(),
                                   nn.Linear(hid_dim[1], 1), nn.Softplus())

    def forward(self, Z):
        mean = self.E_MLP(Z)
        sigma = self.Q_MLP(Z)
        return mean, sigma

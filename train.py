import torch
from itertools import chain
from model import MQENet
import torch.optim as optim
import torch.nn as nn


def train(dataset, args):
    device = torch.device(args.device if args.cuda else 'cpu')
    model_list = []
    X_list = dataset.X_list
    for v in range(len(X_list)):
        feat_dim = X_list[v].shape[1]
        model_temp = MQENet(args.z_dim, feat_dim, args.hid_dim).to(device)
        model_list.append(model_temp)

    Z = torch.normal(mean=torch.zeros([dataset.num_nodes, args.z_dim]), std=0.01)
    if torch.cuda.is_available():
        dataset.y = dataset.y.cuda()
        Z = Z.cuda().detach()
    Z.requires_grad_(True)
    model_list = nn.ModuleList(model_list)
    op = optim.Adam(chain(model_list.parameters(), [Z]), lr=args.lr)
    for epoch in range(1, args.epochs + 1):
        loss = 0
        model_list.train()
        for v in range(len(X_list)):
            x_re, sigma = model_list[v](Z)
            re_loss = (x_re - X_list[v]) ** 2
            re_loss = re_loss.div(2 * sigma ** 2) + torch.log(sigma)
            re_loss = re_loss.mean(1, keepdim=True)
            loss = loss + re_loss.mean()
        op.zero_grad()
        loss.backward()
        op.step()
    return Z.detach().cpu().numpy()
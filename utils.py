from scipy.sparse import csc_matrix
from sklearn.preprocessing import MinMaxScaler
import torch_geometric
import numpy as np
from scipy import sparse as sp
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize
import torch
import os.path as osp
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, WikiCS
from torch_geometric.utils import remove_self_loops, add_self_loops


def load_data(name):
    path = osp.join('./', 'data')
    if name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(path, name)
    elif name in ['computers', 'photo']:
        dataset = Amazon(path, name)
    elif name in ['cs', 'physics']:
        dataset = Coauthor(path, name)
    elif name in ['wikics']:
        dataset = WikiCS(path)

    data = dataset[0]
    data.edge_index = remove_self_loops(data.edge_index)[0]
    data.edge_index = add_self_loops(data.edge_index)[0]
    data.num_classes = torch.max(data.y).item() + 1
    data.x = Norm(data.x)
    adj = edge_index_to_sparse_mx(data.edge_index, data.num_nodes)
    data.A = adj
    adj = process_adj(adj)
    data.adj = adj
    return data


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def set_seeds(seed):
    np.random.seed(seed)
    torch_geometric.seed_everything(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def Norm(x, min=0):
    x = x.detach().cpu().numpy()
    if min == 0:
        scaler = MinMaxScaler((0, 1))
    else:
        scaler = MinMaxScaler((-1, 1))
    norm_x = torch.tensor(scaler.fit_transform(x))
    if torch.cuda.is_available():
        norm_x = norm_x.cuda()
    return norm_x


def MessagePro(data, norm_A, K):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    norm_A = norm_A.to(device)
    X_list = [data]
    for _ in range(K):
        X_list.append(torch.spmm(norm_A, X_list[-1]))

    return X_list


def label_classification(X, Y):
    X = normalize(X, norm='l2')
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.8)

    logreg = LogisticRegression(solver='liblinear')
    c = 2.0 ** np.arange(-10, 10)

    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
                       param_grid=dict(estimator__C=c), n_jobs=8, cv=5,
                       verbose=0)
    clf.fit(X_train, y_train)
    y_pred_test = clf.predict(X_test)
    acc_test = accuracy_score(y_test, y_pred_test)

    return acc_test * 100


def edge_index_to_sparse_mx(edge_index, num_nodes):
    edge_weight = np.array([1] * len(edge_index[0]))
    adj = csc_matrix((edge_weight, (edge_index[0], edge_index[1])),
                     shape=(num_nodes, num_nodes)).tolil()
    return adj


def normalize_adj(adj):
    # Add self-loops
    adj = adj + sp.eye(adj.shape[0])
    # Compute degree matrix
    rowsum = np.array(adj.sum(1))
    # Compute D^{-1/2}
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    # Compute D^{-1/2}AD^{-1/2}
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


def process_adj(adj):
    adj.setdiag(1)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize_adj(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj

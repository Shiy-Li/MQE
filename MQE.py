import torch
import argparse
from utils import *
from train import train
from sklearn.neighbors import kneighbors_graph
import warnings
import numpy as np
import json


def get_knn_graph(x, num_neighbor, knn_metric='cosine'):
    adj_knn = kneighbors_graph(x, num_neighbor, metric=knn_metric)
    return adj_knn.tolil()


def parameter_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=250, help='the train epochs')
    parser.add_argument('--lr', type=float, default=5e-3, metavar='LR', help='learning rate')
    parser.add_argument('--z-dim', type=list, default=256, help='the dim of Z')
    parser.add_argument('--hid-dim', type=int, default=[512, 64], help='the hidden layer dim for E_MLP&Q_MLP')
    parser.add_argument('--data_name', type=str, default='cora', help='the dataset')
    parser.add_argument('--num-class', type=int, default=0, help='the num_class')
    parser.add_argument('--num-hops', type=int, default=8)
    parser.add_argument('--model', type=str, default='MQE')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_neighbor', type=int, default=50)
    parser.add_argument('--alpha', type=float, default=0., help='noisy node frac')
    parser.add_argument('--beta', type=float, default=0., help='noise level')
    parser.add_argument('--knn-metric', type=str, default='cosine', choices=['cosine', 'minkowski'])
    args = parser.parse_known_args()[0]
    return args


def polluted_feat(x, alpha, noise_type, beta):
    rnd_generator = np.random.RandomState(0)
    data_tmp = x.cpu().numpy()
    print('original', data_tmp.sum())
    num_sample, num_feat = data_tmp.shape[0], data_tmp.shape[1]
    num_noisy_samples = int(alpha * num_sample)
    noisy_indices = rnd_generator.choice(num_sample, num_noisy_samples, replace=False)
    clean_indices = np.setdiff1d(np.arange(num_sample), noisy_indices)
    print('noisy_indices', len(noisy_indices))
    print('clean_indices', len(clean_indices))
    if noise_type == 'uniform':
        print('uniform noise !', beta)
        noise = np.random.uniform(0, 1, data_tmp.shape)
    elif noise_type == 'normal':
        print('normal noise !', beta)
        noise = np.random.normal(0, 1, data_tmp.shape)
    data_tmp[noisy_indices] += beta * noise[noisy_indices]
    print('pollted', data_tmp.sum())
    return data_tmp, noisy_indices, clean_indices


def load_best_params(model, dataset, alpha, noise_type, beta, path='./best_params/'):
    save_file = f'best_results_{model}_{dataset}.json'
    file_path = path + save_file
    try:
        with open(file_path, 'r') as f:
            results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Error: Unable to load results from {file_path}")
        return None
    key_str = f"{alpha}_{beta}_{noise_type}"
    if model in results and key_str in results[model]:
        best_params = results[model][key_str]['best_params']
        return best_params
    else:
        print(f"No best params found for model '{model}', dataset '{dataset}', alpha '{alpha}', noise_type '{noise_type}', beta '{beta}'")
        return None


def main(model_name, ntrials, data_name, beta, noise_type, alpha):
    # Load best_params.yaml
    best_params = load_best_params(model_name, data_name, alpha, noise_type, beta)
    if not best_params:
        print('Use the default params')
        # If matching parameters are not found, use default values
        lr, epochs, h1_dim, h2_dim, z_dim, num_hops, num_k = 0.002, 800, 512, 32, 1024, 16, 40
    else:
        print('Use the saved params')
        # Use the parameters from the YAML file
        lr, epochs, h1_dim, h2_dim, z_dim, num_hops, num_k = best_params
        print(best_params)
    warnings.filterwarnings("ignore")
    set_seeds(0)
    args = parameter_parser()
    args.lr, args.epochs, args.z_dim, args.num_hops, args.num_neighbor = lr, epochs, z_dim, num_hops, num_k
    args.hid_dim = [h1_dim, h2_dim]
    args.ntrials = ntrials
    if torch.cuda.is_available():
        args.cuda = True
    else:
        args.cuda = False
    device = torch.device(args.device if args.cuda else 'cpu')
    args.data_name = data_name
    args.painting = False
    args.beta = beta
    args.alpha = alpha
    args.noise_type = noise_type

    dataset = load_data(args.data_name)
    dataset.ori_x = dataset.x.cpu().clone()
    if args.noise_type != 'Original':
        data_tmp, noisy_indices, clean_indices = polluted_feat(dataset.x, alpha, noise_type, beta)
        data_tmp = torch.from_numpy(data_tmp)
        dataset.x = data_tmp
        args.noise_idx = noisy_indices
        args.clean_idx = clean_indices
    dataset.x = dataset.x.to(device)
    feat_ = MessagePro(dataset.x, dataset.adj, args.num_hops)
    feat = torch.stack(feat_).sum(dim=0)
    adj_f = get_knn_graph(feat.cpu(), args.num_neighbor, knn_metric=args.knn_metric)
    adj_f = process_adj(adj_f)
    x_f = MessagePro(dataset.x, (dataset.adj+adj_f)/2, args.num_hops)
    dataset.X_list = x_f
    args.input_dim = dataset.num_node_features
    args.num_class = dataset.num_classes
    args.save_model = False
    args.is_val = True
    accs = []
    for trial in range(1, args.ntrials+1):
        args.trial = trial
        set_seeds(trial)
        H = train(dataset, args)
        Y = dataset.y.detach().cpu().numpy()
        acc_test = label_classification(H, Y)
        print(f'ACC Test = {acc_test:.2f}')
        accs.append(acc_test)

    avg_acc = round(np.mean(accs), 2)
    std_acc = round(np.std(accs), 2)
    return avg_acc, std_acc


warnings.filterwarnings("ignore")
data_name = 'Cora'
beta = 0.8
alpha = 0.5
ntrials = 5
noise_type = 'normal'
model = 'MQE'
main(model, ntrials, data_name, beta, noise_type, alpha)

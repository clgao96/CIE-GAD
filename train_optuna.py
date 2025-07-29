import torch
import torch.nn as nn
import numpy as np
import warnings
import argparse
import optuna
from utils import setup_seed, evaluation_model_prediction, graph_to_normadj, dual_hypergraph_trans
from JacobiHGNN import JacobiHGNN
from data_loader import load_data

warnings.filterwarnings("ignore")
torch.cuda.set_device(2)

setup_seed(2023)

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--hid_dim', type=int, default=32, help='Number of hidden units.')
parser.add_argument('--nlayers', type=int, default=1, help='the number of HGNN layers')
parser.add_argument('--patience', type=int, default=100, help='Patience for early stopping')
parser.add_argument('--ntrials', type=int, default=5, help='Number of trials')
parser.add_argument("--num_classes", type=int, default=2, help="number of class")
parser.add_argument('--train_ratio', type=float, default=0.4)
parser.add_argument('--normalization', type=str, default='sym')
parser.add_argument('--dataset', type=str, default='yelp', help='See choices', choices=['amazon', 'yelp', 'reddit', 'tfinance', 'tsocial', 'elliptic'])
parser.add_argument('--train_save', type=str, default='Jacobi_Hyper', help='model')
parser.add_argument('--model', type=str, default='Jacobiatt3_Hyperhalf_ds', help='model modification')
parser.add_argument("--order", type=int, default=10, help="the order of polynomial")
parser.add_argument("--homo", type=int, default=1, help="1 for Homo and 0 for Hetero")
parser.add_argument('--path', type=str, default="")
parser.add_argument('--optruns', type=int, default=50)
args = parser.parse_args()

print(args)

features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, graph, idx_train, idx_valid, idx_test = load_data(args)

print('train/dev/test samples: ', train_mask.sum().item(), val_mask.sum().item(), test_mask.sum().item())
print('strart generate parse adjacency matrix')

adj, edge_index = graph_to_normadj(graph, labels.size()[0], args.dataset, args.homo)

graph = graph.to(device)
features = features.to(device)
labels_cuda = labels.to(device)
if args.homo:
    adj = adj.to(device)
    edge_index = edge_index.to(device)
    print('strart generate dual hypergraph matrix')
    hg = dual_hypergraph_trans(edge_index, labels_cuda.size(0), features)
    print('finished generate')
else:
    adj = [a.to(device) for a in adj]
    edge_index = [e.to(device) for e in edge_index]
    print('strart generate dual hypergraph matrix')
    hg = [dual_hypergraph_trans(e, labels_cuda.size(0), features) for e in edge_index]
    print('finished generate')


def work(alpha: float = 0.2,
         a: float = 1.0,
         b: float = 1.0,
         drop1: float = 0.0,
         drop2: float = 0.0,
         lr1: float = 1e-3,
         lr2: float = 1e-3,
         wd1: float = 0,
         wd2: float = 0):
    outs = []
    for i in range(args.ntrials):
        model = JacobiHGNN(nfeats, args.hid_dim, args.num_classes, adj, args.order, args.nlayers,
                           alpha, a, b, drop1, drop2).to(device)

        optimizer = torch.optim.Adam([{
            'params': model.params1,
            'weight_decay': wd1,
            'lr': lr1
        }, {
            'params': model.params2,
            'weight_decay': wd2,
            'lr': lr2
        }])

        xloss = nn.CrossEntropyLoss().to(device)
        best_loss=100
        min_val_loss = 100
        max_val_auc = 0.
        counter = 0

        for e in range(args.epochs):
            model.train()
            logit, logit1 = model(hg, features.float())
            loss0 = xloss(logit[train_mask], labels_cuda[train_mask])
            loss1 = xloss(logit1[train_mask], labels_cuda[train_mask])
            loss = loss0 + loss1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.eval()
            loss_val = xloss(logit[val_mask], labels_cuda[val_mask])
            probs = logit.softmax(1).detach().cpu().numpy()
            val_res = evaluation_model_prediction(probs[val_mask], labels[val_mask].numpy())

            if loss_val < min_val_loss and max_val_auc < val_res.auc:
                min_val_loss = loss_val
                max_val_auc = val_res.auc
                counter = 0
            else:
                counter += 1
            if counter >= args.patience:
                print('early stop')
                break
        print('round {}, Val AUC{}'.format(i + 1, val_res.auc))
        outs.append(val_res.auc)
    return np.average(outs)

def search_hyper_params(trial: optuna.Trial):
    lr1 = trial.suggest_categorical("lr1", [0.0005, 0.001, 0.005, 0.01, 0.05])
    lr2 = trial.suggest_categorical("lr2", [0.0005, 0.001, 0.005, 0.01, 0.05])
    wd1 = trial.suggest_categorical("wd1", [0.0, 5e-5, 1e-4, 5e-4, 1e-3])
    wd2 = trial.suggest_categorical("wd2", [0.0, 5e-5, 1e-4, 5e-4, 1e-3])
    alpha = trial.suggest_float('alpha', 0.5, 2.0, step=0.5)
    a = trial.suggest_float('a', -1.0, 2.0, step=0.25)
    b = trial.suggest_float('b', -0.5, 2.0, step=0.25)
    drop1 = trial.suggest_float("drop1", 0.0, 0.3, step=0.1)
    drop2 = trial.suggest_float("drop2", 0.0, 0.3, step=0.1)
    return work(alpha, a, b, drop1, drop2, lr1, lr2, wd1, wd2)

study = optuna.create_study(direction="maximize",
                            study_name=args.dataset + '_h{}_o{}'.format(args.hid_dim, args.order),
                            load_if_exists=False)
study.optimize(search_hyper_params, n_trials=args.optruns)
print("best params ", study.best_params)
print("best valauc ", study.best_value)
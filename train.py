import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
import argparse
import os
import time
import dgl
from utils import setup_seed, evaluation_model_prediction, graph_to_normadj, dual_hypergraph_trans
from JacobiHGNN import JacobiHGNN
from data_loader import load_data


warnings.filterwarnings("ignore")
torch.cuda.set_device(0)

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


setup_seed(2023)

def nll_loss(pred, target, pos_w: float=1.0):
    weight_tensor = torch.tensor([1., pos_w]).to(pred.device)
    loss_value = F.nll_loss(pred, target.long(), weight=weight_tensor)

    return loss_value

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--lr1', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--lr2', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--wd1', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--wd2', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hid_dim', type=int, default=32, help='Number of hidden units.')
parser.add_argument('--nlayers', type=int, default=1, help='the number of HGNN layers')
parser.add_argument('--patience', type=int, default=150, help='Patience for early stopping')
parser.add_argument('--ntrials', type=int, default=10, help='Number of trials')
parser.add_argument("--num_classes", type=int, default=2, help="number of class")
parser.add_argument('--train_ratio', type=float, default=0.4)
parser.add_argument('--normalization', type=str, default='sym')
parser.add_argument('--dataset', type=str, default='yelp', help='See choices', choices=['amazon', 'yelp', 'reddit', 'tfinance', 'tsocial', 'elliptic'])
parser.add_argument('--train_save', type=str, default='Jacobi_Hyper', help='model')
parser.add_argument('--model', type=str, default='Jacobiatt3_Hyperhalf', help='model modification')
parser.add_argument("--homo", type=int, default=1, help="1 for Homo and 0 for Hetero")
parser.add_argument("--order", type=int, default=10, help="the order of polynomial")
parser.add_argument("--alpha", type=float, default= 2.0, help="the parameter to initialize polynomial coefficients")
parser.add_argument("--a", type=float, default= -0.25, help="the initialize jacobi coefficients")
parser.add_argument("--b", type=float, default= 2.0, help="the initialize jacobi coefficients")
parser.add_argument('--drop1', type=float, default=0.0, help='Dropout rate of JacobiGNN (1 - keep probability).')
parser.add_argument('--drop2', type=float, default=0.0, help='Dropout rate of HGNN (1 - keep probability).')
parser.add_argument('--gamma', type=float, default=1.0, help='γ')
args = parser.parse_args()
print(args)

features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, graph, idx_train, idx_valid, idx_test = load_data(args)

print('train/dev/test samples: ', train_mask.sum().item(), val_mask.sum().item(), test_mask.sum().item())

sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
input_nodes, output_nodes, blocks = sampler.sample_blocks(graph, idx_train)

print('strart generate parse adjacency matrix')
adj, edge_index = graph_to_normadj(graph, labels.size()[0], args.dataset, args.homo)
print('finished generate')

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


model_path = './snapshots/{}/'.format(args.dataset)
os.makedirs(model_path, exist_ok=True)
loss_path = './train_loss/{}/'.format(args.dataset)
os.makedirs(loss_path, exist_ok=True)
result_path = './results/{}/'.format(args.dataset)
os.makedirs(result_path, exist_ok=True)

argsDict = args.__dict__
with open(loss_path + 'setting.txt', 'w') as f:
    f.writelines('------------------ start ------------------' + '\n')
    for eachArg, value in argsDict.items():
        f.writelines(eachArg + ' : ' + str(value) + '\n')
    f.writelines('------------------- end -------------------')

final_aucs, final_aps, final_mf1s, final_recs, final_pres = [], [], [], [], []
performance = np.zeros((args.ntrials, 5))

for i in range(args.ntrials):
    print("#" * 20, "Start Training round{}".format(i), "#" * 20)
    model = JacobiHGNN(nfeats, args.hid_dim, args.num_classes, adj, depth=args.order, nlayers=args.nlayers,
                       alpha=args.alpha, a=args.a, b=args.b, drop1=args.drop1, drop2=args.drop2).to(device)

    param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {param}")
    optimizer = torch.optim.Adam([
        {'params': model.params1, 'weight_decay': args.wd1, 'lr': args.lr1},
        {'params': model.params2, 'weight_decay': args.wd2, 'lr': args.lr2},
    ])
    xloss = nn.CrossEntropyLoss().to(device)
    best_f1, final_tf1, final_trec, final_tpre, final_tmf1, final_tauc = 0., 0., 0., 0., 0., 0.
    best_loss, min_val_loss = 100, 100
    best_auc, max_val_auc = 0., 0.
    best_val = None
    counter = 0

    time_start = time.time()

    for e in range(args.epochs):
        model.train()
        logit, logit1 = model(hg, features.float())

        loss0 = xloss(logit[train_mask], labels_cuda[train_mask])
        loss1 = xloss(logit1[train_mask], labels_cuda[train_mask])
        loss = loss0 + args.gamma * loss1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        loss_val = xloss(logit[val_mask], labels_cuda[val_mask])
        logit, logit1 = model(hg, features.float())
        probs = logit.softmax(1).detach().cpu().numpy()
        val_res = evaluation_model_prediction(probs[val_mask], labels[val_mask].numpy())

        if loss <= best_loss:
            best_loss = loss
            best_val = val_res
            torch.save(model.state_dict(), model_path + '{}.pth'.format(i))
            print("epoch {}, Val Loss {:.4f}, Val Auc {:.4f}, Val AP {:.4f}, Val macro_F1 {:.4F}, Val recall {:.4F}, Val precision {:.4F}".format(
                e, loss_val, best_val.auc, best_val.ap, best_val.macro_F1, best_val.recall, best_val.precision))
            test_res = evaluation_model_prediction(probs[test_mask], labels[test_mask].numpy())
            print("epoch {}, Test Auc {:.4f}, Test AP {:.4f}, Test macro_F1 {:.4F}, Test Recall {:.4F}, Test Precision {:.4F}".format(
                e, test_res.auc, test_res.ap, test_res.macro_F1, test_res.recall, test_res.precision))
        # early stop
        if loss_val < min_val_loss and max_val_auc < val_res.auc:
            min_val_loss = loss_val
            max_val_auc = val_res.auc
            counter = 0
        else:
            counter += 1
        if counter >= args.patience:
            print('early stop')
            break

    time_end = time.time()
    print('time cost: ', time_end - time_start, 's')
    print('Trial{},Test: AUC {:.2f} AP {:.2f} MF1 {:.2f} REC {:.2f} PRE {:.2f} '.format(i, test_res.auc * 100, test_res.ap * 100, test_res.macro_F1 * 100, test_res.recall * 100, test_res.precision * 100))

    final_aucs.append(test_res.auc)
    final_aps.append(test_res.ap)
    final_mf1s.append(test_res.macro_F1)
    final_recs.append(test_res.recall)
    final_pres.append(test_res.precision)

print('AUC-mean: {:.2f}, AUC-std: {:.2f}, AP-mean: {:.2f}, AP-std: {:.2f}, MF1-mean: {:.2f}, MF1-std: {:.2f}, '
      'Recall-mean: {:.2f}, Recall-std: {:.2f}, Prec-mean: {:.2f}, Prec-std: {:.2f}'.format(
    100 * np.mean(final_aucs), 100 * np.std(final_aucs),
    100 * np.mean(final_aps), 100 * np.std(final_aps),
    100 * np.mean(final_mf1s), 100 * np.std(final_mf1s),
    100 * np.mean(final_recs), 100 * np.std(final_recs),
    100 * np.mean(final_pres), 100 * np.std(final_pres)))

print(args)

performance = np.stack([np.array(final_aucs), np.array(final_aps), np.array(final_mf1s), np.array(final_recs), np.array(final_pres)], axis=1)
np.savetxt(result_path +'{}.txt'.format(args.model), performance)




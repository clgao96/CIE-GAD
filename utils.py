import torch
import dgl
import numpy as np
import scipy.sparse as sp
import random
import math
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, roc_auc_score, average_precision_score
from collections import namedtuple
from torch_geometric.utils import degree
from torch_sparse import SparseTensor



def setup_seed(random_seed):
    r"""Sets the seed for generating random numbers."""
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    
def normalize_features(mx, norm_row=True):
    """
    Row-normalize sparse matrix
        Code from https://github.com/williamleif/graphsage-simple/
    """

    if norm_row:
        rowsum = np.array(mx.sum(1)) + 0.01
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)

    else:
        column_max = mx.max(dim=0)[0].unsqueeze(0)
        column_min = mx.min(dim=0)[0].unsqueeze(0)
        min_max_column_norm = (mx - column_min) / (column_max - column_min)
        # l2_norm = torch.norm(min_max_column_norm, p=2, dim=-1, keepdim=True)
        mx = min_max_column_norm
    return mx


def graph_to_normadj(graph, n_node, data, homo):
    """
    one graph
    convert dgl.Graph to edge_index and edge_weight to the sparse adjacency matrix.

    """
    if homo:
        adj = graph.adjacency_matrix()
        edge_index = adj._indices()  # (Tensor): shape (2, number of edges)
        edge_weight = adj._values()  # (Tensor): shape (number of edges)
        deg = degree(edge_index[0], n_node)
        deg[deg < 0.5] += 1.0
        deg = torch.pow(deg, -0.5)  # d^(-1/2)
        val = deg[edge_index[0]] * edge_weight * deg[edge_index[1]]
        ret = SparseTensor(row=edge_index[0],
                           col=edge_index[1],
                           value=val,
                           sparse_sizes=(n_node, n_node)).coalesce()

    # elif homo == 0 and data == 'yelp':
    #     print(graph.canonical_etypes)
    #     adj_1 = graph.adjacency_matrix(etype='net_rsr')
    #     adj_2 = graph.adjacency_matrix(etype='net_rtr')
    #     adj_3 = graph.adjacency_matrix(etype='net_rur')
    #
    #     edge_index_1 = adj_1._indices()  # (Tensor): shape (2, number of edges)
    #     edge_weight_1 = adj_1._values()  # (Tensor): shape (number of edges)
    #     deg_1 = degree(edge_index_1[0], n_node)
    #     deg_1[deg_1 < 0.5] += 1.0
    #     deg_1 = torch.pow(deg_1, -0.5)  # d^(-1/2)
    #     val_1 = deg_1[edge_index_1[0]] * edge_weight_1 * deg_1[edge_index_1[1]]
    #     ret_1 = SparseTensor(row=edge_index_1[0], col=edge_index_1[1], value=val_1,
    #                          sparse_sizes=(n_node, n_node)).coalesce()
    #
    #     edge_index_2 = adj_2._indices()  # (Tensor): shape (2, number of edges)
    #     edge_weight_2 = adj_2._values()  # (Tensor): shape (number of edges)
    #     deg_2 = degree(edge_index_2[0], n_node)
    #     deg_2[deg_2 < 0.5] += 1.0
    #     deg_2 = torch.pow(deg_2, -0.5)  # d^(-1/2)
    #     val_2 = deg_2[edge_index_2[0]] * edge_weight_2 * deg_2[edge_index_2[1]]
    #     ret_2 = SparseTensor(row=edge_index_2[0], col=edge_index_2[1], value=val_2,
    #                          sparse_sizes=(n_node, n_node)).coalesce()
    #
    #     edge_index_3 = adj_3._indices()  # (Tensor): shape (2, number of edges)
    #     edge_weight_3 = adj_3._values()  # (Tensor): shape (number of edges)
    #     deg_3 = degree(edge_index_3[0], n_node)
    #     deg_3[deg_3 < 0.5] += 1.0
    #     deg_3 = torch.pow(deg_3, -0.5)  # d^(-1/2)
    #     val_3 = deg_3[edge_index_3[0]] * edge_weight_3 * deg_3[edge_index_3[1]]
    #     ret_3 = SparseTensor(row=edge_index_3[0], col=edge_index_3[1], value=val_3,
    #                          sparse_sizes=(n_node, n_node)).coalesce()
    #
    #     ret = [ret_1, ret_2, ret_3]
    #     edge_index = [edge_index_1, edge_index_2, edge_index_3]
    #
    # elif homo == 0 and data == 'amazon':
    #     print(graph.etypes)
    #     adj_1 = graph.adjacency_matrix(etype='net_upu')
    #     adj_2 = graph.adjacency_matrix(etype='net_usu')
    #     adj_3 = graph.adjacency_matrix(etype='net_uvu')
    #
    #     edge_index_1 = adj_1._indices()  # (Tensor): shape (2, number of edges)
    #     edge_weight_1 = adj_1._values()  # (Tensor): shape (number of edges)
    #     deg_1 = degree(edge_index_1[0], n_node)
    #     deg_1[deg_1 < 0.5] += 1.0
    #     deg_1 = torch.pow(deg_1, -0.5)  # d^(-1/2)
    #     val_1 = deg_1[edge_index_1[0]] * edge_weight_1 * deg_1[edge_index_1[1]]
    #     ret_1 = SparseTensor(row=edge_index_1[0], col=edge_index_1[1], value=val_1,
    #                          sparse_sizes=(n_node, n_node)).coalesce()
    #
    #     edge_index_2 = adj_2._indices()  # (Tensor): shape (2, number of edges)
    #     edge_weight_2 = adj_2._values()  # (Tensor): shape (number of edges)
    #     deg_2 = degree(edge_index_2[0], n_node)
    #     deg_2[deg_2 < 0.5] += 1.0
    #     deg_2 = torch.pow(deg_2, -0.5)  # d^(-1/2)
    #     val_2 = deg_2[edge_index_2[0]] * edge_weight_2 * deg_2[edge_index_2[1]]
    #     ret_2 = SparseTensor(row=edge_index_2[0], col=edge_index_2[1], value=val_2,
    #                          sparse_sizes=(n_node, n_node)).coalesce()
    #
    #     edge_index_3 = adj_3._indices()  # (Tensor): shape (2, number of edges)
    #     edge_weight_3 = adj_3._values()  # (Tensor): shape (number of edges)
    #     deg_3 = degree(edge_index_3[0], n_node)
    #     deg_3[deg_3 < 0.5] += 1.0
    #     deg_3 = torch.pow(deg_3, -0.5)  # d^(-1/2)
    #     val_3 = deg_3[edge_index_3[0]] * edge_weight_3 * deg_3[edge_index_3[1]]
    #     ret_3 = SparseTensor(row=edge_index_3[0], col=edge_index_3[1], value=val_3,
    #                          sparse_sizes=(n_node, n_node)).coalesce()
    #
    #     ret = [ret_1, ret_2, ret_3]
    #     edge_index = [edge_index_1, edge_index_2, edge_index_3]

    else:
        print('This data is homo=1.')

    return ret, edge_index


class HGObject:
    pass

def dual_hypergraph_trans(edge_index, n_node, features): #
    # adjacency matrix of graph -> incidence matrix of graph
    num_edge = edge_index.size(1)
    col = torch.arange(0,num_edge,1).repeat_interleave(2).view(1,-1).squeeze().to(edge_index.device)
    row = edge_index.T.reshape(1,-1).squeeze().to(edge_index.device)
    val = torch.ones(row.size(0)).to(edge_index.device)

    MT = SparseTensor(row=col, col=row, value=val, sparse_sizes=(num_edge, n_node)).coalesce()
    # node degree, edge degree of hypergraph
    D_e = MT.sum(0) # 
    D_v = MT.sum(1) # sum(W*MT, dim=1)
    D_e = torch.pow(D_e, -0.5)
    D_e1 = torch.pow(D_e, -1)
    D_v = torch.pow(D_v, -0.5)
    D_v1 = torch.pow(D_v, -1)
    row_e = col_e = torch.arange(D_e.size(0), dtype=torch.long).to(edge_index.device)
    row_v = col_v = torch.arange(D_v.size(0), dtype=torch.long).to(edge_index.device)
    D_e = SparseTensor(row=row_e, col=col_e, value=D_e, sparse_sizes=(D_e.size(0), D_e.size(0))).coalesce()
    D_e1 = SparseTensor(row=row_e, col=col_e, value=D_e1, sparse_sizes=(D_e.size(0), D_e.size(0))).coalesce()
    D_v = SparseTensor(row=row_v, col=col_v, value=D_v, sparse_sizes=(D_v.size(0), D_v.size(0))).coalesce()
    D_v1 = SparseTensor(row=row_v, col=col_v, value=D_v1, sparse_sizes=(D_v.size(0), D_v.size(0))).coalesce()
    
    hg = HGObject()
    hg.MT = MT
    hg.D_e = D_e
    hg.D_e1 = D_e1
    hg.D_v = D_v
    hg.D_v1 =D_v1

    return hg


Evaluation_Metrics = namedtuple('Evaluation_Metrics', ['acc', 'macro_F1', 'recall', 'precision', 'auc', 'ap', 'gmean'])

def gmean(y_true, y_pred):
    """binary geometric mean of  True Positive Rate (TPR) and True Negative Rate (TNR)

    Args:
            y_true (np.array): label
            y_pred (np.array): prediction
    """

    TP, TN, FP, FN = 0, 0, 0, 0
    for sample_true, sample_pred in zip(y_true, y_pred):
        TP += sample_true * sample_pred
        TN += (1 - sample_true) * (1 - sample_pred)
        FP += (1 - sample_true) * sample_pred
        FN += sample_true * (1 - sample_pred)

    return math.sqrt(TP * TN / (TP + FN) / (TN + FP))

def evaluation_model_prediction(pred_logit, label):
    pred_label = np.argmax(pred_logit, axis=1)
    pred_logit = pred_logit[:, 1]

    accuracy = accuracy_score(label, pred_label)
    f1 = f1_score(label, pred_label, average='macro')
    recall = recall_score(label, pred_label, average='macro')
    precision = precision_score(label, pred_label, average='macro')
    auc = roc_auc_score(label, pred_logit)
    ap = average_precision_score(label, pred_logit, average='macro')
    gmean_value = gmean(label, pred_label)

    return Evaluation_Metrics(acc=accuracy, macro_F1=f1, recall=recall, precision=precision, auc=auc, ap=ap, gmean=gmean_value)


    
def to_edge_index(adj):
    if isinstance(adj, torch.Tensor):
        row, col, value = adj.to_sparse_coo().indices()[0], adj.to_sparse_coo().indices()[1], \
                        adj.to_sparse_coo().values()

    elif isinstance(adj, sp.csr_matrix):
        row, col, value = adj.tocoo().row, adj.tocoo().col, \
                          adj.tocoo().data
        row, col, value = torch.tensor(row, dtype=torch.long), torch.tensor(col, dtype=torch.long), \
                          torch.tensor(value, dtype=torch.float)
    else:
        raise RuntimeError("adj has to be either torch.sparse_csr_matrix or scipy.sparse.csr_matrix.")
    if value is None:
        value = torch.ones(row.size(0), device=row.device)

    return torch.stack([row, col], dim=0), value
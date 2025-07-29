import warnings
import dgl
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from utils import normalize_features
from dgl.data import FraudYelpDataset, FraudAmazonDataset
from dgl.data.utils import load_graphs
import pickle

warnings.simplefilter("ignore")


def load_data(args):
    dataset_str = args.dataset

    if dataset_str == 'yelp':
        dataset = FraudYelpDataset()
        graph = dataset[0]
        if args.homo:
            graph = dgl.to_homogeneous(graph, ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
            graph = dgl.add_self_loop(graph)
        elif args.homo == 0:
            for etype in ['net_rsr', 'net_rur', 'net_rtr']:
                graph = dgl.add_self_loop(graph, etype=etype)

        train_mask, val_mask, test_mask, idx_train, idx_valid, idx_test = graph_split(dataset_str, graph.ndata['label'],
                                                                                      train_ratio=args.train_ratio,
                                                                                      folds=args.ntrials)

        graph.ndata['feature'] = torch.tensor(normalize_features(graph.ndata['feature'], norm_row=False),
                                              dtype=torch.float)

        return graph.ndata['feature'], graph.ndata['feature'].size()[-1], graph.ndata['label'], 2, \
            train_mask, val_mask, test_mask, graph, idx_train, idx_valid, idx_test

    elif dataset_str == 'amazon':
        dataset = FraudAmazonDataset()
        graph = dataset[0]
        if args.homo:
            graph = dgl.to_homogeneous(graph, ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
            graph = dgl.add_self_loop(graph)
        elif args.homo == 0:
            for etype in ['net_upu', 'net_usu', 'net_uvu']:
                graph = dgl.add_self_loop(graph, etype=etype)

        train_mask, val_mask, test_mask, idx_train, idx_valid, idx_test = graph_split(dataset_str, graph.ndata['label'],
                                                                                      train_ratio=args.train_ratio,
                                                                                      folds=args.ntrials)

        graph.ndata['feature'] = torch.tensor(normalize_features(graph.ndata['feature'], norm_row=True),
                                              dtype=torch.float)

        return graph.ndata['feature'], graph.ndata['feature'].size()[-1], graph.ndata['label'], 2, \
            train_mask, val_mask, test_mask, graph, idx_train, idx_valid, idx_test


    elif dataset_str == 'tfinance':
        graph, label_dict = load_graphs('./data/tfinance')
        graph = graph[0]
        graph = dgl.add_self_loop(graph)
        graph.ndata['label'] = graph.ndata['label'].argmax(1)

        train_mask, val_mask, test_mask, idx_train, idx_valid, idx_test = graph_split(dataset_str, graph.ndata['label'],
                                                                                      train_ratio=args.train_ratio,
                                                                                      folds=args.ntrials)

        return graph.ndata['feature'], graph.ndata['feature'].size()[-1], graph.ndata['label'], 2, \
            train_mask, val_mask, test_mask, graph, idx_train, idx_valid, idx_test

            
    elif dataset_str == 'elliptic':
        data = pickle.load(open('./data/{}.dat'.format('elliptic'), 'rb'))
        graph = dgl.graph((data.edge_index[0], data.edge_index[1]))
        graph = dgl.add_self_loop(graph)
        graph.ndata['feature'] = data.x
        graph.ndata['label'] = data.y.type(torch.LongTensor)

        train_mask, val_mask, test_mask, idx_train, idx_valid, idx_test = graph_split(dataset_str, graph.ndata['label'],
                                                                                      train_ratio=args.train_ratio,
                                                                                      folds=args.ntrials)

        graph.ndata['feature'] = torch.tensor(normalize_features(graph.ndata['feature'], norm_row=True),
                                              dtype=torch.float)

        return graph.ndata['feature'], graph.ndata['feature'].size()[-1], graph.ndata['label'], 2, \
            train_mask, val_mask, test_mask, graph, idx_train, idx_valid, idx_test

    else:
        raise NotImplementedError


def graph_split(dataset, labels, train_ratio=0.4, folds=5):
    """split dataset into train and test

    Args:
        dataset (str): name of dataset
        labels (list): list of labels of nodes
    """
    assert dataset in ['amazon', 'yelp', 'reddit', 'tfinance', 'tsocial', 'dgraphfin', 'elliptic']
    if dataset == 'amazon':
        index = np.array(range(3305, len(labels)))
        stratify_labels = labels[3305:]

    elif dataset == 'yelp' or dataset == 'reddit' or dataset == 'tfinance' or dataset == 'tsocial' or dataset == 'elliptic':
        index = np.array(range(len(labels)))
        stratify_labels = labels

    else:
        index = np.array(range(46564))
        stratify_labels = labels[:46564]

    # generate mask
    idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels[index], stratify=labels[index],
                                                            train_size=train_ratio, random_state=2, shuffle=True)
    idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest, test_size=0.67,
                                                            random_state=2, shuffle=True)

    train_mask = torch.zeros([len(labels)]).bool()
    valid_mask = torch.zeros([len(labels)]).bool()
    test_mask = torch.zeros([len(labels)]).bool()

    train_mask[idx_train] = 1
    valid_mask[idx_valid] = 1
    test_mask[idx_test] = 1
    return train_mask, valid_mask, test_mask, idx_train, idx_valid, idx_test



import time
import warnings, os
import argparse
from utils import read_pickle, DrugDataset2, DrugDataLoader, CustomData
import pandas as pd
from model import nnModel_AUC_FC
import numpy as np
from loss import InfoNCE
from tqdm import tqdm
import torch.optim as optim
import torch
import torch.nn as nn
from metric import mse,rmse,ci,pearson,spearman

def test_nn(model, test_loader,device):
    model.eval()
    pred_list = []
    label_list = []

    total_loss_test = 0

    with torch.no_grad():

        for idx, data in enumerate(tqdm(test_loader, mininterval=0.5, desc='Evaluating', leave=False, ncols=50)):
            head_pairs, tail_pairs, label = [d.to(device) for d in data]
            pred, _, _, _, _ = model((head_pairs, tail_pairs))
            # pred_cls = torch.sigmoid(pred)
            pred_list.append(pred.detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())

    pred_probs = np.concatenate(pred_list, axis=0).flatten()
    label = np.concatenate(label_list, axis=0)

    ret_test = [rmse(label, pred_probs), mse(label, pred_probs), pearson(label, pred_probs),
                spearman(label, pred_probs), ci(label, pred_probs)]
    msg1 = "RMSE-%.4f,MSE-%.4f,Pearson-%.4f,Spearman-%.4f,CI-%.4f" % (ret_test[0], ret_test[1], ret_test[2], ret_test[3], ret_test[4])

    print(msg1)
    print()

    return ret_test[0],ret_test[1],ret_test[2],ret_test[3],ret_test[4]

def test_MMDDI(args):
    root = 'data/preprocessed/' + args.dataset
    batch_size = args.batch_size

    drug_graph = read_pickle(os.path.join(root, 'drug_data.pkl'))
    drug_graph = read_pickle(os.path.join(root, 'drug_data.pkl'))

    fold1 = pd.read_csv(os.path.join(root, f'fold1.csv'), delimiter=',')
    fold1['Y'] = fold1['Y'].apply(lambda x: np.log2(x))

    fold2 = pd.read_csv(os.path.join(root, f'fold2.csv'), delimiter=',')
    fold2['Y'] = fold2['Y'].apply(lambda x: np.log2(x))

    fold3 = pd.read_csv(os.path.join(root, f'fold3.csv'), delimiter=',')
    fold3['Y'] = fold3['Y'].apply(lambda x: np.log2(x))

    fold4 = pd.read_csv(os.path.join(root, f'fold4.csv'), delimiter=',')
    fold4['Y'] = fold4['Y'].apply(lambda x: np.log2(x))

    fold5 = pd.read_csv(os.path.join(root, f'fold5.csv'), delimiter=',')
    fold5['Y'] = fold5['Y'].apply(lambda x: np.log2(x))

    all_folds = [fold1, fold2, fold3, fold4, fold5]

    results = []

    for i in range(5):
        test_df = all_folds[i]
        print('#' * 15, 'fold ', i + 1, '#' * 15)
        test_set = DrugDataset2(test_df, drug_graph, args)
        test_loader = DrugDataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

        data = next(iter(test_loader))

        node_dim = data[0].x.shape[1]
        edge_dim = 6  # data[0].edge_attr.size(-1)

        device = torch.device('cuda:' + args.device)

        model = nnModel_AUC_FC(node_dim, edge_dim, n_iter=args.n_iter, args=args).to(device)

        model_path = root + '/model'+str(i+1)+'/model.pkl'

        model.load_state_dict(torch.load(model_path, map_location=device))


        rmse,mse,pearson,spearman,ci = test_nn(model=model, test_loader=test_loader, device=device)
        torch.cuda.empty_cache()

        results.append([round(rmse,4),round(mse,4),round(pearson,4),round(spearman,4),round(ci,4)])

    results = np.vstack(results)
    results_mean = np.mean(results, axis=0)  # .tolist()
    results_std = np.std(results, axis=0)  # .tolist()
    print('\t'.join([str(item) for item in results_mean]))
    print('\t'.join([str(item) for item in results_std]))


def test_External(args):
    root = 'data/preprocessed/' + args.dataset
    batch_size = args.batch_size

    drug_graph = read_pickle(os.path.join(root, 'drug_data.pkl'))

    external = pd.read_csv(os.path.join(root, f'External.csv'), delimiter=',')
    external['Y'] = external['Y'].apply(lambda x: np.log2(x))

    results = []

    for i in range(5):
        print('#' * 15, 'fold ', i + 1, '#' * 15)
        external_set = DrugDataset2(external, drug_graph, args)
        external_loader = DrugDataLoader(external_set, batch_size=batch_size, shuffle=False,
                                         num_workers=args.num_workers)

        data = next(iter(external_loader))

        node_dim = data[0].x.shape[1]
        edge_dim = 6  # data[0].edge_attr.size(-1)

        device = torch.device('cuda:' + args.device)

        model = nnModel_AUC_FC(node_dim, edge_dim, n_iter=args.n_iter, args=args).to(device)

        model_path = root + '/model'+str(i+1)+'/model_External.pkl'

        model.load_state_dict(torch.load(model_path, map_location=device))


        rmse,mse,pearson,spearman,ci = test_nn(model=model, test_loader=external_loader, device=device)
        torch.cuda.empty_cache()

        results.append([round(rmse,4),round(mse,4),round(pearson,4),round(spearman,4),round(ci,4)])

    results = np.vstack(results)
    results_mean = np.mean(results, axis=0)  # .tolist()
    results_std = np.std(results, axis=0)  # .tolist()
    print('\t'.join([str(item) for item in results_mean]))
    print('\t'.join([str(item) for item in results_std]))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument('--n_iter', type=int, default=10, help='number of iterations')
    parser.add_argument('--epochs', type=int, default=120, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--save_model', action='store_true', help='whether save model or not')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--dataset', type=str, default='AUC_FC', help='AUC_FC')
    parser.add_argument('--log', type=str, default=0, help='logging or not.')
    parser.add_argument('--device', type=str, default='1', help='cuda device.')
    parser.add_argument('--num_workers', type=int, default=8, help='num_workers.')
    parser.add_argument('--heads', type=int, default=2, help='heads.')
    args = parser.parse_args()
    print(args)
    test_MMDDI(args)
    test_External(args)
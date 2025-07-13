import time
import warnings,os
import argparse
from utils import read_pickle,split_train_valid,DrugDataset1,DrugDataset2,DrugDataLoader,CustomData
import pandas as pd
import torch
from model import nnModel,nnModel2
from trainNN import train_nn_warm,train_nn_cold

def train(args):

    root = 'data/preprocessed/'+args.dataset

    fold = args.fold
    batch_size = args.batch_size

    drug_graph = read_pickle(os.path.join(root, 'drug_data.pkl'))

    if args.split == 'warm' and (args.dataset == 'drugbank' or args.dataset == 'twosides'):
        train_df = pd.read_csv(os.path.join(root,f'pair_pos_neg_triplets_train_fold{fold}.csv'))
        test_df = pd.read_csv(os.path.join(root, f'pair_pos_neg_triplets_test_fold{fold}.csv'))
        train_df, val_df = split_train_valid(train_df, fold=fold)
        train_set = DrugDataset1(train_df, drug_graph, args)
        val_set = DrugDataset1(val_df, drug_graph, args)
        test_set = DrugDataset1(test_df, drug_graph, args)

    elif args.dataset == 'ZhangDDI' or args.dataset == 'ChChMiner' or args.dataset == 'DeepDDI':
        train_df = pd.read_csv(os.path.join(root, args.dataset+'_train.csv'))
        val_df = pd.read_csv(os.path.join(root, args.dataset+'_valid.csv'))
        test_df = pd.read_csv(os.path.join(root, args.dataset+'_test.csv'))
        train_set = DrugDataset2(train_df, drug_graph, args)
        val_set = DrugDataset2(val_df, drug_graph, args)
        test_set = DrugDataset2(test_df, drug_graph, args)
    elif args.split == 'cold' and args.dataset == 'drugbank':
        train_df = pd.read_csv(os.path.join(root, f'cold/fold0/train.csv'))
        val_df = pd.read_csv(os.path.join(root, f'cold/fold0/s1.csv'))
        test_df = pd.read_csv(os.path.join(root, f'cold/fold0/s2.csv'))
        train_set = DrugDataset1(train_df, drug_graph, args)
        val_set = DrugDataset1(val_df, drug_graph, args)
        test_set = DrugDataset1(test_df, drug_graph, args)
    else:
        raise Exception('Error Input for --split and --dataset')



    train_loader = DrugDataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)#drop_last=True?
    valid_loader = DrugDataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DrugDataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)



    data = next(iter(train_loader))
    if args.dataset=='drugbank':
        node_dim = 70  # data[0].x.size(-1)
    elif args.dataset == 'twosides':
        node_dim = 54
    elif args.dataset == 'ZhangDDI':
        node_dim = 45
    elif args.dataset == 'ChChMiner':
        node_dim = 54
    elif args.dataset == 'DeepDDI':
        node_dim = 71
    edge_dim = 6  # data[0].edge_attr.size(-1)

    device = torch.device('cuda:'+args.device)
    if args.dataset == 'drugbank' or args.dataset == 'twosides':
        model = nnModel(node_dim, edge_dim, n_iter=args.n_iter,args=args).to(device)
    else:
        model = nnModel2(node_dim, edge_dim, n_iter=args.n_iter, args=args).to(device)

    if args.split == 'warm':
        train_nn_warm(model=model, train_loader=train_loader, valid_loader = valid_loader,test_loader=test_loader, device=device, args=args)
    else:
        train_nn_cold(model=model, train_loader=train_loader,valid_loader=valid_loader, test_loader=test_loader,
                      device=device, args=args)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument('--n_iter', type=int, default=10, help='number of iterations')
    parser.add_argument('--fold', type=int, default=0, help='[0, 1, 2]')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--save_model', action='store_true', help='whether save model or not')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--dataset', type=str, default='drugbank', help='drugbank, twosides,ZhangDDI,ChChMiner, or DeepDDI')
    parser.add_argument('--split', type=str, default='warm', help='warm or cold.')
    # parser.add_argument('--split2', type=str, default='s1', help='s1 or s2 for cold.')
    parser.add_argument('--log', type=str, default=0, help='logging or not.')
    parser.add_argument('--device', type=str, default='1', help='cuda device.')
    parser.add_argument('--num_workers', type=int, default=6, help='num_workers.')
    parser.add_argument('--heads', type=int, default=2, help='heads.')
    args = parser.parse_args()
    print(args)
    train(args)
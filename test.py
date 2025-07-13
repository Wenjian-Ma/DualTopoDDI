import warnings,os
import argparse
from utils import read_pickle,split_train_valid,DrugDataset1,DrugDataset2,DrugDataLoader,CustomData
import pandas as pd
import torch,numpy as np
from tqdm import tqdm
from model import nnModel,nnModel2
from metric import do_compute_metrics
# from trainNN import train_nn_warm,train_nn_cold



def test_nn_warm(model, test_loader, device, args):
    model.eval()
    pred_list = []
    label_list = []

    with torch.no_grad():
        for idx, data in enumerate(tqdm(test_loader, mininterval=0.5, desc='Evaluating', leave=False, ncols=50)):
            if args.dataset == 'drugbank' or args.dataset=='twosides':
                head_pairs, tail_pairs, rel, label = [d.to(device) for d in data]
                pred, _, _, _, _ = model((head_pairs, tail_pairs, rel))
            else:
                head_pairs, tail_pairs, label = [d.to(device) for d in data]
                pred, _, _, _, _  = model((head_pairs, tail_pairs))


            # pred = model((head_pairs, tail_pairs, rel))
            ###################

            ####################
            # pred_cls = torch.sigmoid(pred)
            pred_list.append(pred.view(-1).detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())

    pred_probs = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    acc, auroc, f1_score, precision, recall, ap = do_compute_metrics(pred_probs, label)

    msg2 = "Test_acc-%.4f, Test_auroc-%.4f, Test_f1_score-%.4f, Test_prec-%.4f, Test_rec-%.4f, Test_ap-%.4f" % (
         acc, auroc, f1_score, precision, recall, ap)

    print(msg2)
    print()
def test_nn_cold(model, valid_loader, test_loader,device, args):
    model.eval()
    pred_list = []
    label_list = []



    with torch.no_grad():

        for idx, data in enumerate(tqdm(valid_loader, mininterval=0.5, desc='Evaluating', leave=False, ncols=50)):
            head_pairs, tail_pairs, rel, label = [d.to(device) for d in data]

            pred, _, _, _, _ = model((head_pairs, tail_pairs, rel))


            # pred_cls = torch.sigmoid(pred)
            pred_list.append(pred.view(-1).detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())

    pred_probs = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    acc, auroc, f1_score, precision, recall, ap = do_compute_metrics(pred_probs, label)

    msg1 = " S1_acc-%.4f, S1_auroc-%.4f, S1_f1_score-%.4f, S1_prec-%.4f, S1_rec-%.4f, S1_ap-%.4f" % (
         acc, auroc, f1_score, precision, recall, ap)

    print(msg1)

    #####################################
    model.eval()
    pred_list = []
    label_list = []



    with torch.no_grad():

        for idx, data in enumerate(tqdm(test_loader, mininterval=0.5, desc='Evaluating', leave=False, ncols=50)):
            head_pairs, tail_pairs, rel, label = [d.to(device) for d in data]

            pred, _, _, _, _ = model((head_pairs, tail_pairs, rel))


            # pred_cls = torch.sigmoid(pred)
            pred_list.append(pred.view(-1).detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())

    pred_probs = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    acc, auroc, f1_score, precision, recall, ap = do_compute_metrics(pred_probs, label)



    msg2 = " S2_acc-%.4f, S2_auroc-%.4f, S2_f1_score-%.4f, S2_prec-%.4f, S2_rec-%.4f, S2_ap-%.4f" % (
         acc, auroc, f1_score, precision, recall, ap)

    print(msg2)
    print()



def test(args):

    root = 'data/preprocessed/'+args.dataset

    fold = args.fold
    batch_size = args.batch_size

    drug_graph = read_pickle(os.path.join(root, 'drug_data.pkl'))

    if args.split == 'warm' and (args.dataset == 'drugbank' or args.dataset == 'twosides'):
        test_df = pd.read_csv(os.path.join(root, f'pair_pos_neg_triplets_test_fold{fold}.csv'))
        test_set = DrugDataset1(test_df, drug_graph, args)
        test_loader = DrugDataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
    elif args.dataset == 'ZhangDDI' or args.dataset == 'ChChMiner' or args.dataset == 'DeepDDI':
        test_df = pd.read_csv(os.path.join(root, args.dataset+'_test.csv'))
        test_set = DrugDataset2(test_df, drug_graph, args)
        test_loader = DrugDataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
    elif args.split == 'cold' and args.dataset == 'drugbank':

        val_df = pd.read_csv(os.path.join(root, f'cold/fold0/s1.csv'))
        val_set = DrugDataset1(val_df, drug_graph, args)
        valid_loader = DrugDataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

        test_df = pd.read_csv(os.path.join(root, f'cold/fold0/s2.csv'))
        test_set = DrugDataset1(test_df, drug_graph, args)
        test_loader = DrugDataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
    else:
        raise Exception('Error Input for --split and --dataset')

    if args.dataset == 'drugbank':
        node_dim = 70  # data[0].x.size(-1)
    elif args.dataset == 'twosides':
        node_dim = 54
        args.heads = 1
    elif args.dataset == 'ZhangDDI':
        node_dim = 45
    elif args.dataset == 'ChChMiner':
        node_dim = 54
    elif args.dataset == 'DeepDDI':
        node_dim = 71
    edge_dim = 6  # data[0].edge_attr.size(-1)

    device = torch.device('cuda:'+args.device)

    if args.dataset == 'drugbank' or args.dataset == 'twosides':
        model = nnModel(node_dim, edge_dim, n_iter=args.n_iter, args=args).to(device)
    else:
        model = nnModel2(node_dim, edge_dim, n_iter=args.n_iter, args=args).to(device)

    if args.split == 'warm':

        model_path = root + '/model/'+str(args.fold)+'/model.pkl'

        model.load_state_dict(torch.load(model_path, map_location=device))

        test_nn_warm(model=model, test_loader=test_loader, device=device, args=args)
    else:
        model_path = root + '/cold_model/'+str(args.fold)+'/model.pkl'
        model.load_state_dict(torch.load(model_path, map_location=device))

        test_nn_cold(model=model, valid_loader=valid_loader, test_loader=test_loader,
                      device=device, args=args)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument('--n_iter', type=int, default=10, help='number of iterations')
    parser.add_argument('--fold', type=int, default=0, help='[0, 1, 2]')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--dataset', type=str, default='drugbank', help='drugbank, twosides,ZhangDDI,ChChMiner, or DeepDDI')
    parser.add_argument('--split', type=str, default='warm', help='warm or cold.')
    parser.add_argument('--log', type=str, default=0, help='logging or not.')
    parser.add_argument('--device', type=str, default='1', help='cuda device.')
    parser.add_argument('--num_workers', type=int, default=6, help='num_workers.')
    parser.add_argument('--heads', type=int, default=2, help='heads.')
    args = parser.parse_args()
    print(args)
    test(args)
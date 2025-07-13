import time
import warnings, os
import argparse
from utils import read_pickle, DrugDataset3, DrugDataLoader, CustomData
import pandas as pd
from model import nnModel_MMDDI
import numpy as np
from loss import InfoNCE
from tqdm import tqdm
import torch.optim as optim
import torch
import torch.nn as nn
from metric import do_compute_metrics_MMDDI
import torch.nn.functional as F

def train_nn(model, train_loader, test_loader, external_loader, device, args, fold):
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,
                                              total_steps=args.epochs * len(train_loader)) 

    for epoch in range(args.epochs):
        total_loss = 0
        model.train()
        for idx, data in enumerate(tqdm(train_loader, mininterval=0.5, desc='Training', leave=False, ncols=50)):
            head_pairs, tail_pairs, label = [d.to(device) for d in data]
            pred, h_g_node_list, h_g_line_list, t_g_node_list, t_g_line_list = model((head_pairs, tail_pairs))

            cl_loss = (InfoNCE(h_g_node_list, h_g_line_list, device=device) + InfoNCE(t_g_node_list, t_g_line_list,
                                                                                      device=device))

            ce_loss = criterion(pred, label)

            loss = ce_loss + 0.4 * cl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss = total_loss + ce_loss

        total_loss = total_loss / (idx + 1)

        #########################
        model.eval()
        pred_list = []
        label_list = []
        total_loss_test = 0

        with torch.no_grad():

            for idx, data in enumerate(tqdm(test_loader, mininterval=0.5, desc='Evaluating', leave=False, ncols=50)):
                head_pairs, tail_pairs, label = [d.to(device) for d in data]
                pred, _, _, _, _ = model((head_pairs, tail_pairs))
                loss = criterion(pred, label)

                pred = F.softmax(pred,dim=-1)

                # pred_cls = torch.sigmoid(pred)
                pred_list.append(pred.detach().cpu().numpy())
                label_list.append(label.detach().cpu().numpy())

                total_loss_test = total_loss_test + loss

        pred_probs = np.concatenate(pred_list, axis=0)
        label = np.concatenate(label_list, axis=0)

        total_loss_test = total_loss_test / (idx + 1)
        acc1, auroc1, f1_score1, precision1, recall1, ap1 = do_compute_metrics_MMDDI(pred_probs, label)

        msg1 = "Epoch-%d, Train_loss-%.4f, Test_loss-%.4f, Test_acc-%.4f, Test_auroc-%.4f, Test_f1_score-%.4f, Test_prec-%.4f, Test_rec-%.4f, Test_ap-%.4f" % (
            epoch + 1, total_loss, total_loss_test, acc1, auroc1, f1_score1, precision1, recall1, ap1)

        print(msg1)

        #########################
        model.eval()
        pred_list = []
        label_list = []

        total_loss_test = 0

        with torch.no_grad():

            for idx, data in enumerate(
                    tqdm(external_loader, mininterval=0.5, desc='Evaluating', leave=False, ncols=50)):
                head_pairs, tail_pairs, label = [d.to(device) for d in data]
                pred, _, _, _, _ = model((head_pairs, tail_pairs))
                loss = criterion(pred, label)

                pred = F.softmax(pred, dim=-1)

                # pred_cls = torch.sigmoid(pred)
                pred_list.append(pred.detach().cpu().numpy())
                label_list.append(label.detach().cpu().numpy())

                total_loss_test = total_loss_test + loss

        pred_probs = np.concatenate(pred_list, axis=0)
        label = np.concatenate(label_list, axis=0)
        total_loss_test = total_loss_test / (idx + 1)
        acc2, auroc2, f1_score2, precision2, recall2, ap2 = do_compute_metrics_MMDDI(pred_probs, label)

        msg2 = "Epoch-%d, Train_loss-%.4f, Test_loss-%.4f, Test_acc-%.4f, Test_auroc-%.4f, Test_f1_score-%.4f, Test_prec-%.4f, Test_rec-%.4f, Test_ap-%.4f" % (
            epoch + 1, total_loss, total_loss_test, acc2, auroc2, f1_score2, precision2, recall2, ap2)

        print(msg2)
        print()

        if args.log:
            with open('./log/' + str(args) + '.txt', 'a') as f:
                f.write(msg2 + '\n\n')
        if args.save_model:
            torch.save(model.state_dict(),
                       './data/preprocessed/' + args.dataset + '/model' + str(fold + 1) + '/Epoch_' + str(
                           epoch) + '_' + str(acc1) + '_' + str(acc2) + '.pkl')


def train(args):
    root = 'data/preprocessed/' + args.dataset
    batch_size = args.batch_size

    drug_graph = read_pickle(os.path.join(root, 'drug_data.pkl'))

    fold1 = pd.read_csv(os.path.join(root, f'fold1.csv'), delimiter=',')
    fold2 = pd.read_csv(os.path.join(root, f'fold2.csv'), delimiter=',')
    fold3 = pd.read_csv(os.path.join(root, f'fold3.csv'), delimiter=',')
    fold4 = pd.read_csv(os.path.join(root, f'fold4.csv'), delimiter=',')
    fold5 = pd.read_csv(os.path.join(root, f'fold5.csv'), delimiter=',')

    DDInter = pd.read_csv(os.path.join(root, f'DDInter.csv'), delimiter=',')

    for i in range(5):

        all_folds = [fold1, fold2, fold3, fold4, fold5]

        print('#' * 15, 'fold ', i + 1, '#' * 15)

        test_df = all_folds[i]
        all_folds.pop(i)
        train_df = pd.concat(all_folds)

        train_set = DrugDataset3(train_df, drug_graph, args)
        test_set = DrugDataset3(test_df, drug_graph, args)
        external_set = DrugDataset3(DDInter, drug_graph, args)

        train_loader = DrugDataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=args.num_workers,
                                      drop_last=False) 
        test_loader = DrugDataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
        external_loader = DrugDataLoader(external_set, batch_size=batch_size, shuffle=False,
                                         num_workers=args.num_workers)

        data = next(iter(train_loader))

        node_dim = data[0].x.shape[1]
        edge_dim = 6  # data[0].edge_attr.size(-1)

        device = torch.device('cuda:' + args.device)

        model = nnModel_MMDDI(node_dim, edge_dim, n_iter=args.n_iter, args=args).to(device)

        train_nn(model=model, train_loader=train_loader, test_loader=test_loader, external_loader=external_loader,
                 device=device, args=args, fold=i)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument('--n_iter', type=int, default=10, help='number of iterations')
    parser.add_argument('--epochs', type=int, default=120, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--save_model', action='store_true', help='whether save model or not')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--dataset', type=str, default='MMDDI', help='MMDDI')
    parser.add_argument('--log', type=str, default=0, help='logging or not.')
    parser.add_argument('--device', type=str, default='0', help='cuda device.')
    parser.add_argument('--num_workers', type=int, default=8, help='num_workers.')
    parser.add_argument('--heads', type=int, default=2, help='heads.')
    args = parser.parse_args()
    print(args)
    train(args)

import warnings,os
import argparse
from utils import read_pickle,split_train_valid,DrugDataset1,DrugDataset2,DrugDataLoader,CustomData
import pandas as pd
import torch
from model import nnModel,nnModel2
import torch.optim as optim
import torch
import torch.nn as nn
from metric import accuracy,do_compute_metrics
from tqdm import tqdm
import numpy as np
from loss import InfoNCE



def train_nn_warm(model,train_loader,valid_loader, test_loader,device,args,lambda1,lambda2,fold):

    criterion = nn.BCELoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.epochs * len(train_loader))#480   960   1919 22882

    for epoch in range(args.epochs):
        total_acc=0
        total_loss=0
        total_cl_loss = 0
        model.train()
        for idx,data in enumerate(tqdm(train_loader,mininterval=0.5,desc='Training',leave=False,ncols=50)):
            if args.dataset == 'drugbank' or args.dataset=='twosides':
                head_pairs, tail_pairs, rel, label = [d.to(device) for d in data]
                pred, h_g_node_list, h_g_line_list, t_g_node_list, t_g_line_list = model((head_pairs, tail_pairs, rel))
            else:
                head_pairs, tail_pairs, label = [d.to(device) for d in data]
                pred, h_g_node_list, h_g_line_list, t_g_node_list, t_g_line_list = model((head_pairs, tail_pairs))

            cl_loss = (InfoNCE(h_g_node_list,h_g_line_list,device=device)+InfoNCE(t_g_node_list,t_g_line_list,device=device))

            bce_loss = criterion(pred, label)

            loss = lambda1*bce_loss + lambda2*cl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            pred_cls = (pred > 0.5).detach().cpu().numpy()
            acc = accuracy(label.detach().cpu().numpy(), pred_cls)
            total_acc = total_acc + acc
            total_loss = total_loss + bce_loss
            total_cl_loss = total_cl_loss + cl_loss
        total_acc = total_acc/(idx+1)
        total_loss = total_loss/(idx+1)
        total_cl_loss = total_cl_loss/(idx+1)

        model.eval()
        pred_list = []
        label_list = []

        total_loss_test = 0

        with torch.no_grad():

            for idx,data in enumerate(tqdm(valid_loader,mininterval=0.5,desc='Evaluating',leave=False,ncols=50)):
                if args.dataset == 'drugbank' or args.dataset == 'twosides':
                    head_pairs, tail_pairs, rel, label = [d.to(device) for d in data]
                    pred, _, _, _, _ = model((head_pairs, tail_pairs, rel))
                else:
                    head_pairs, tail_pairs, label = [d.to(device) for d in data]
                    pred, _, _, _, _ = model((head_pairs, tail_pairs))

                loss = criterion(pred, label)

                pred_list.append(pred.view(-1).detach().cpu().numpy())
                label_list.append(label.detach().cpu().numpy())

                total_loss_test = total_loss_test + loss

        pred_probs = np.concatenate(pred_list, axis=0)
        label = np.concatenate(label_list, axis=0)
        total_loss_test = total_loss_test/(idx+1)
        acc, auroc, f1_score, precision, recall, ap = do_compute_metrics(pred_probs, label)

        msg1 = "Epoch-%d, Train_loss-%.4f, Train_acc-%.4f, Val__loss-%.4f, Val__acc-%.4f, Val__auroc-%.4f, Val__f1_score-%.4f, Val__prec-%.4f, Val__rec-%.4f, Val__ap-%.4f" % (
        epoch+1, total_loss, total_acc, total_loss_test, acc, auroc, f1_score, precision, recall, ap)

        print(msg1)
        
        model.eval()
        pred_list = []
        label_list = []

        total_loss_test = 0

        with torch.no_grad():

            for idx,data in enumerate(tqdm(test_loader,mininterval=0.5,desc='Evaluating',leave=False,ncols=50)):
                if args.dataset == 'drugbank' or args.dataset == 'twosides':
                    head_pairs, tail_pairs, rel, label = [d.to(device) for d in data]
                    pred, _, _, _, _ = model((head_pairs, tail_pairs, rel))
                else:
                    head_pairs, tail_pairs, label = [d.to(device) for d in data]
                    pred, _, _, _, _ = model((head_pairs, tail_pairs))

                loss = criterion(pred, label)

                # pred_cls = torch.sigmoid(pred)
                pred_list.append(pred.view(-1).detach().cpu().numpy())
                label_list.append(label.detach().cpu().numpy())

                total_loss_test = total_loss_test + loss

        pred_probs = np.concatenate(pred_list, axis=0)
        label = np.concatenate(label_list, axis=0)
        total_loss_test = total_loss_test/(idx+1)
        acc, auroc, f1_score, precision, recall, ap = do_compute_metrics(pred_probs, label)

        msg2 = "Epoch-%d, Train_loss-%.4f, Train_acc-%.4f, Test_loss-%.4f, Test_acc-%.4f, Test_auroc-%.4f, Test_f1_score-%.4f, Test_prec-%.4f, Test_rec-%.4f, Test_ap-%.4f" % (
        epoch+1, total_loss,total_acc, total_loss_test, acc, auroc, f1_score, precision, recall, ap)

        print(msg2)
        print()
        if args.log:
            with open('./log/'+str(lambda1)+'_'+str(lambda2)+'_fold'+str(fold)+'.txt','a') as f:
                f.write(msg1 + '\n')
                f.write(msg2+'\n\n')
        if args.save_model:
            torch.save(model.state_dict(),'./data/preprocessed/'+args.dataset+'/model/'+str(args.fold)+'/Epoch_'+str(epoch)+'_'+str(acc)+'.pkl')








def train(args):

    root = 'data/preprocessed/'+args.dataset
    drug_graph = read_pickle(os.path.join(root, 'drug_data.pkl'))
    batch_size = args.batch_size
    for lambda1 in [0.4]:
        for lambda2 in [1]:
            for fold in [1,2]:
                # if not (lambda1==0.4 and lambda2==1):
                #     continue
                if args.split == 'warm' and (args.dataset == 'drugbank' or args.dataset == 'twosides'):
                    train_df = pd.read_csv(os.path.join(root,f'pair_pos_neg_triplets_train_fold{fold}.csv'))
                    test_df = pd.read_csv(os.path.join(root, f'pair_pos_neg_triplets_test_fold{fold}.csv'))
                    train_df, val_df = split_train_valid(train_df, fold=fold)
                    train_set = DrugDataset1(train_df, drug_graph, args)
                    val_set = DrugDataset1(val_df, drug_graph, args)
                    test_set = DrugDataset1(test_df, drug_graph, args)

                train_loader = DrugDataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)#drop_last=True?
                valid_loader = DrugDataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
                test_loader = DrugDataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

                data = next(iter(train_loader))
                if args.dataset=='drugbank':
                    node_dim = 70  # data[0].x.size(-1)
                edge_dim = 6  # data[0].edge_attr.size(-1)

                device = torch.device('cuda:'+args.device)
                if args.dataset == 'drugbank' or args.dataset == 'twosides':
                    model = nnModel(node_dim, edge_dim, n_iter=args.n_iter,args=args).to(device)

                if args.split == 'warm':
                    train_nn_warm(model=model, train_loader=train_loader, valid_loader = valid_loader,test_loader=test_loader, device=device, args=args,lambda1=lambda1,lambda2=lambda2,fold=fold)

                torch.cuda.empty_cache()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument('--n_iter', type=int, default=5, help='number of iterations')
    parser.add_argument('--fold', type=int, default=0, help='[0, 1, 2]')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--save_model', action='store_true', help='whether save model or not')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--dataset', type=str, default='drugbank', help='drugbank, twosides,ZhangDDI,ChChMiner, or DeepDDI')
    parser.add_argument('--split', type=str, default='warm', help='warm or cold.')
    # parser.add_argument('--split2', type=str, default='s1', help='s1 or s2 for cold.')
    parser.add_argument('--log', type=str, default=1, help='logging or not.')
    parser.add_argument('--device', type=str, default='1', help='cuda device.')
    parser.add_argument('--num_workers', type=int, default=6, help='num_workers.')
    parser.add_argument('--heads', type=int, default=1, help='heads.')
    args = parser.parse_args()
    print(args)
    train(args)

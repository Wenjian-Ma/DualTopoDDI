import time

import torch.optim as optim
import torch
import torch.nn as nn
from metric import accuracy,do_compute_metrics
from tqdm import tqdm
import numpy as np
from loss import InfoNCE

def train_nn_warm(model,train_loader,valid_loader, test_loader,device,args):

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
            # pred = model((head_pairs, tail_pairs, rel))


            ###################

            cl_loss = (InfoNCE(h_g_node_list,h_g_line_list,device=device)+InfoNCE(t_g_node_list,t_g_line_list,device=device))
            ###################


            bce_loss = criterion(pred, label)

            ###################
            loss = bce_loss + 0.2*cl_loss#0.4
            ####################


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
        ######################
        # model.eval()
        # pred_list = []
        # label_list = []
        #
        # total_loss_test = 0
        #
        # with torch.no_grad():
        #
        #     for idx,data in enumerate(tqdm(valid_loader,mininterval=0.5,desc='Evaluating',leave=False,ncols=50)):
        #         if args.dataset == 'drugbank' or args.dataset == 'twosides':
        #             head_pairs, tail_pairs, rel, label = [d.to(device) for d in data]
        #             pred, _, _, _, _ = model((head_pairs, tail_pairs, rel))
        #         else:
        #             head_pairs, tail_pairs, label = [d.to(device) for d in data]
        #             pred, _, _, _, _ = model((head_pairs, tail_pairs))
        #
        #         ###################
        #
        #         ####################
        #
        #
        #         loss = criterion(pred, label)
        #
        #         # pred_cls = torch.sigmoid(pred)
        #         pred_list.append(pred.view(-1).detach().cpu().numpy())
        #         label_list.append(label.detach().cpu().numpy())
        #
        #         total_loss_test = total_loss_test + loss
        #
        # pred_probs = np.concatenate(pred_list, axis=0)
        # label = np.concatenate(label_list, axis=0)
        # total_loss_test = total_loss_test/(idx+1)
        # acc, auroc, f1_score, precision, recall, ap = do_compute_metrics(pred_probs, label)
        #
        # msg1 = "Epoch-%d, Train_loss-%.4f, Train_acc-%.4f, Val__loss-%.4f, Val__acc-%.4f, Val__auroc-%.4f, Val__f1_score-%.4f, Val__prec-%.4f, Val__rec-%.4f, Val__ap-%.4f" % (
        # epoch+1, total_loss, total_acc, total_loss_test, acc, auroc, f1_score, precision, recall, ap)
        #
        # print(msg1)
        #########################
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

                #pred = model((head_pairs, tail_pairs, rel))

                ###################

                ####################


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
            with open('./log/'+str(args)+'.txt','a') as f:
                f.write(msg2+'\n\n')
        if args.save_model:
            torch.save(model.state_dict(),'./data/preprocessed/'+args.dataset+'/model/'+str(args.fold)+'/Epoch_'+str(epoch)+'_'+str(acc)+'.pkl')


def train_nn_cold(model,train_loader, valid_loader,test_loader,device,args):

    criterion = nn.BCELoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.epochs * len(train_loader))


    for epoch in range(args.epochs):
        total_acc = 0
        total_loss = 0
        total_cl_loss = 0
        model.train()
        for idx,data in enumerate(tqdm(train_loader,mininterval=0.5,desc='Training',leave=False,ncols=50)):
            head_pairs, tail_pairs, rel, label = [d.to(device) for d in data]

            pred, h_g_node_list, h_g_line_list, t_g_node_list, t_g_line_list = model((head_pairs, tail_pairs, rel))
            cl_loss = (InfoNCE(h_g_node_list, h_g_line_list, device=device) + InfoNCE(t_g_node_list, t_g_line_list,device=device)) * 0.1#0.1

            bce_loss = criterion(pred, label)

            ###################
            loss = bce_loss + cl_loss
            ####################

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            pred_cls = (pred > 0.5).detach().cpu().numpy()
            acc = accuracy(label.detach().cpu().numpy(), pred_cls)
            total_acc = total_acc + acc
            total_loss = total_loss + bce_loss
            total_cl_loss = total_cl_loss + cl_loss
        total_acc = total_acc / (idx + 1)
        total_loss = total_loss / (idx + 1)


        #############################################
        model.eval()
        pred_list = []
        label_list = []

        total_loss_test = 0

        with torch.no_grad():

            for idx,data in enumerate(tqdm(valid_loader,mininterval=0.5,desc='Evaluating',leave=False,ncols=50)):
                head_pairs, tail_pairs, rel, label = [d.to(device) for d in data]

                pred, _, _, _, _ = model((head_pairs, tail_pairs, rel))
                loss = criterion(pred, label)

                # pred_cls = torch.sigmoid(pred)
                pred_list.append(pred.view(-1).detach().cpu().numpy())
                label_list.append(label.detach().cpu().numpy())

                total_loss_test = total_loss_test + loss

        pred_probs = np.concatenate(pred_list, axis=0)
        label = np.concatenate(label_list, axis=0)
        total_loss_test = total_loss_test/(idx+1)
        acc, auroc, f1_score, precision, recall, ap = do_compute_metrics(pred_probs, label)

        msg1 = "Epoch-%d, Train_loss-%.4f, Train_acc-%.4f, S1_loss-%.4f, S1_acc-%.4f, S1_auroc-%.4f, S1_f1_score-%.4f, S1_prec-%.4f, S1_rec-%.4f, S1_ap-%.4f" % (
        epoch+1, total_loss, total_acc, total_loss_test, acc, auroc, f1_score, precision, recall, ap)

        print(msg1)

        #####################################
        model.eval()
        pred_list = []
        label_list = []

        total_loss_test = 0

        with torch.no_grad():

            for idx, data in enumerate(tqdm(test_loader, mininterval=0.5, desc='Evaluating', leave=False, ncols=50)):
                head_pairs, tail_pairs, rel, label = [d.to(device) for d in data]

                pred, _, _, _, _ = model((head_pairs, tail_pairs, rel))
                loss = criterion(pred, label)

                # pred_cls = torch.sigmoid(pred)
                pred_list.append(pred.view(-1).detach().cpu().numpy())
                label_list.append(label.detach().cpu().numpy())

                total_loss_test = total_loss_test + loss

        pred_probs = np.concatenate(pred_list, axis=0)
        label = np.concatenate(label_list, axis=0)
        total_loss_test = total_loss_test / (idx + 1)
        acc, auroc, f1_score, precision, recall, ap = do_compute_metrics(pred_probs, label)

        msg2 = "Epoch-%d, Train_loss-%.4f, Train_acc-%.4f, S2_loss-%.4f, S2_acc-%.4f, S2_auroc-%.4f, S2_f1_score-%.4f, S2_prec-%.4f, S2_rec-%.4f, S2_ap-%.4f" % (
            epoch + 1, total_loss, total_acc, total_loss_test, acc, auroc, f1_score, precision, recall, ap)

        print(msg2)
        print()

        if args.log:
            with open('./log/' + str(args) + '.txt', 'a') as f:
                f.write( msg2 + '\n\n')

        if args.save_model:
            torch.save(model.state_dict(),'./data/preprocessed/'+args.dataset+'/cold_model/'+str(args.fold)+'/Epoch_'+str(epoch)+'_'+str(acc)+'.pkl')



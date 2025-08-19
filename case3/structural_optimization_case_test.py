import time,sys
import warnings, os
import argparse
sys.path.append('..')
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

    return pred_probs

def test_MMDDI(args):
    drug = 'Paroxetine'#Paroxetine/Itraconazole

    root = '/media/ST-18T/Ma/DualTopoDDI/data/preprocessed/case3'
    batch_size = args.batch_size

    drug_graph = read_pickle(os.path.join(root, 'drug_data.pkl'))

    Paroxetine = pd.read_csv(os.path.join(root, f'Paroxetine.csv'), delimiter=',')
    Itraconazole = pd.read_csv(os.path.join(root, f'Itraconazole.csv'), delimiter=',')

    if drug == 'Paroxetine':
        test_df = Paroxetine
        mark = int(Paroxetine.shape[0] / 2)
    elif drug == 'Itraconazole':
        test_df = Itraconazole
        mark = int(Itraconazole.shape[0] / 4)

    test_set = DrugDataset2(test_df, drug_graph, args)
    test_loader = DrugDataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

    data = next(iter(test_loader))

    node_dim = data[0].x.shape[1]
    edge_dim = 6  # data[0].edge_attr.size(-1)

    device = torch.device('cuda:' + args.device)

    model = nnModel_AUC_FC(node_dim, edge_dim, n_iter=args.n_iter, args=args).to(device)

    model_path = '/media/ST-18T/Ma/DualTopoDDI/data/preprocessed/case3/model.pkl'

    model.load_state_dict(torch.load(model_path, map_location=device))


    pred_probs = test_nn(model=model, test_loader=test_loader, device=device).reshape(-1,1)
    if drug == 'Paroxetine':
        pred_probs = np.hstack([pred_probs[:mark],pred_probs[mark:]])
    else:
        pred_probs = np.hstack([pred_probs[:mark],pred_probs[mark:mark*2],pred_probs[mark*2:mark*3], pred_probs[mark*3:]])
    torch.cuda.empty_cache()


    with open('/media/ST-18T/Ma/DualTopoDDI/data/preprocessed/case3/' + drug + '_result1.csv', 'a') as f:
        for i in range(pred_probs.shape[0]):
            for j in range(pred_probs.shape[1]):
                f.write(str(pred_probs[i][j])+',')
            f.write('\n')


    print(pred_probs)

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


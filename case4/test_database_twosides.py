import warnings,os,sys
import argparse
sys.path.append('..')
import pandas as pd
import torch,numpy as np
from tqdm import tqdm
from model import *
from sklearn.model_selection import StratifiedShuffleSplit
from torch_geometric.data import Batch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data


class nnModel3(torch.nn.Module):
    def __init__(self, in_dim, edge_in_dim, hidden_dim=64, n_iter=10,args=None):
        super(nnModel3, self).__init__()

        self.drug_encoder = DrugEncoder(in_dim, edge_in_dim, hidden_dim, n_iter=n_iter,args=args)

        self.hma = hierarchical_mutual_attn(hidden_dim)

        self.rmodule = nn.Embedding(1316, hidden_dim)

        self.sigmoid = nn.Sigmoid()

        if args.dataset == 'drugbank':
            self.lin = nn.Sequential(
                nn.Linear(hidden_dim * 4, hidden_dim * 8),
                nn.PReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim * 8, hidden_dim * 4),
                nn.PReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim * 4, hidden_dim),
            )
        elif args.dataset == 'twosides':
            self.lin = nn.Sequential(
                nn.Linear(hidden_dim * 4, hidden_dim * 8),
                nn.BatchNorm1d(hidden_dim*8),
                nn.PReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim * 8, hidden_dim * 4),
                nn.BatchNorm1d(hidden_dim*4),
                nn.PReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim * 4, hidden_dim),
            )
    def forward(self, triples):
        h_data, t_data, rels = triples

        x_node_h,x_line_h,h_g_node_list,h_g_line_list = self.drug_encoder(h_data)
        x_node_t, x_line_t,t_g_node_list,t_g_line_list = self.drug_encoder(t_data)

        x_node_h,x_node_t,x_line_h,x_line_t = self.hma(x_node_h,x_node_t,x_line_h,x_line_t,h_data, t_data)

        rep = torch.cat([x_node_h,x_line_h, x_node_t,x_line_t], dim=-1)


        rfeat = self.rmodule(rels)

        logit = (self.lin(rep) * rfeat).sum(-1)

        output = self.sigmoid(logit)

        return output,h_g_node_list,h_g_line_list,t_g_node_list,t_g_line_list

class DrugDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data, collate_fn=data.collate_fn, **kwargs)

class CustomData(Data):
    '''
    Since we have converted the node graph to the line graph, we should specify the increase of the index as well.
    '''
    def __inc__(self, key, value, *args, **kwargs):

        if key == 'line_graph_edge_index':
            return self.edge_index.size(1) if self.edge_index.nelement()!=0 else 0
        return super().__inc__(key, value, *args, **kwargs)

class DrugDataset2(Dataset):
    def __init__(self, data_df, drug_graph,args):
        self.data_df = data_df
        self.drug_graph = drug_graph
        self.args = args

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        return self.data_df.iloc[index]

    def collate_fn(self, batch):
        head_list = []
        tail_list = []
        label_list = []

        for row in batch:
            Drug1_ID, Drug2_ID, Y = row['ID1'], row['ID2'], row['Y']
            h_graph = self.drug_graph.get(Drug1_ID)
            t_graph = self.drug_graph.get(Drug2_ID)

            if (h_graph.x.shape[0]==1 or h_graph.edge_index.shape[0]==0 or h_graph.edge_attr.shape[0]==0 or h_graph.line_graph_edge_index.shape[0]==0) or (t_graph.x.shape[0]==1 or t_graph.edge_index.shape[0]==0 or t_graph.edge_attr.shape[0]==0 or t_graph.line_graph_edge_index.shape[0]==0) :
                continue

            head_list.append(h_graph)
            tail_list.append(t_graph)
            label_list.append(torch.LongTensor([Y]))
        if len(head_list)==0 or len(tail_list)==0:
            return label_list,label_list,label_list
        head_pairs = Batch.from_data_list(head_list, follow_batch=['edge_index','line_graph_edge_index'])
        tail_pairs = Batch.from_data_list(tail_list, follow_batch=['edge_index','line_graph_edge_index'])

        label = torch.cat(label_list, dim=0)
        return head_pairs, tail_pairs, label

def test_nn_warm(model, test_loader, device, args,rel_dict):
    model.eval()

    with torch.no_grad():
        for idx, data in enumerate(tqdm(test_loader, mininterval=0.5, desc='Evaluating', leave=False, ncols=50)):
            if len(data[2]) == 0:
                continue
            head_pairs, tail_pairs, rel = [d.to(device) for d in data]

            if args.dataset == 'drugbank' or args.dataset=='twosides':
                pred, _, _, _, _ = model((head_pairs, tail_pairs, rel))
                rel_list=rel.view(-1).detach().cpu().numpy().tolist()
            else:
                pred, _, _, _, _  = model((head_pairs, tail_pairs))
            pred_list=pred.view(-1).detach().cpu().numpy().tolist()
            h_id=head_pairs.id
            t_id=tail_pairs.id


            if args.dataset == 'drugbank' or args.dataset == 'twosides':
                with open('/media/ST-18T/Ma/DualTopoDDI/data/preprocessed/case4/database/'+args.dataset+'_constructed_database.csv','a') as f:
                    for i in range(len(h_id)):
                        f.write(h_id[i]+','+t_id[i]+','+rel_dict[str(rel_list[i])]+','+str(pred_list[i])+'\n')
            elif args.dataset == 'DeepDDI':
                with open('/media/ST-18T/Ma/DualTopoDDI/data/preprocessed/case4/database/'+args.dataset+'_constructed_database.csv','a') as f:
                    for i in range(len(h_id)):
                        f.write(h_id[i]+','+t_id[i]+','+str(pred_list[i])+'\n')




def get_rel_dict(dataset):
    rel_dict = {}
    with open('/media/ST-18T/Ma/DualTopoDDI/data/preprocessed/case4/'+dataset+'_rel_dict.txt') as f:
        for line in f:
            Y = line.strip().split('\t')[0]
            content = line.strip().split('\t')[1]
            rel_dict[Y] = content
    return rel_dict


def test(args):

    batch_size = args.batch_size

    drug_graph = pd.read_pickle('/media/ST-18T/Ma/DualTopoDDI/data/preprocessed/case4/'+args.dataset+'_drug.pkl')
    approved_drug = {}
    with open('/media/ST-18T/Ma/DualTopoDDI/data/preprocessed/case4/approved_drug.csv') as f:
        for line in f:
            id = line.strip().split(',')[0]
            smiles = line.strip().split(',')[1]
            approved_drug[id] = smiles

    test_df = []
    rel_dict = {}

    node_dim = 88
    edge_dim = 6  # data[0].edge_attr.size(-1)
    args.heads = 1

    device = torch.device('cuda:' + args.device)

    model = nnModel3(node_dim, edge_dim, n_iter=args.n_iter, args=args).to(device)

    model_path = '/media/ST-18T/Ma/DualTopoDDI/data/preprocessed/case4/model/' + args.dataset + '/model.pkl'

    model.load_state_dict(torch.load(model_path, map_location=device))


    rel_dict = get_rel_dict(args.dataset)
    rel_dict_reverse = {j: i for i, j in rel_dict.items()}

    with open('/media/ST-18T/Ma/DualTopoDDI/data/preprocessed/case4/database/twosides_constructed_database.csv') as f:
        for line in f:
            a = line.strip().split(',')[0]
            b = line.strip().split(',')[1]
            c = rel_dict_reverse[line.strip().split(',')[2]]
            d = line.strip().split(',')[3]
    mark = 0

    for i,smiles_i in tqdm(list(approved_drug.items())[800:1600]):
        for j,smiles_j in approved_drug.items():
            for k in range(1316):
                if i == a and j == b and k == int(c):
                    mark = 1
                    continue
                if i!=j and mark == 1:
                    test_df.append([i,j,smiles_i,smiles_j,k])
        if mark == 0:
            continue
        test_df = pd.DataFrame(test_df, columns=['ID1', 'ID2','X1','X2','Y'])


        test_set = DrugDataset2(test_df, drug_graph, args)
        test_loader = DrugDataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

        test_nn_warm(model=model, test_loader=test_loader, device=device, args=args,rel_dict=rel_dict)

        test_df = []

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument('--n_iter', type=int, default=10, help='number of iterations')
    parser.add_argument('--fold', type=int, default=0, help='[0, 1, 2]')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--dataset', type=str, default='twosides', help='drugbank, twosides,or DeepDDI')
    parser.add_argument('--split', type=str, default='warm', help='warm or cold.')
    parser.add_argument('--log', type=str, default=0, help='logging or not.')
    parser.add_argument('--device', type=str, default='0', help='cuda device.')
    parser.add_argument('--num_workers', type=int, default=6, help='num_workers.')
    parser.add_argument('--heads', type=int, default=2, help='heads.')
    args = parser.parse_args()
    print(args)
    test(args)


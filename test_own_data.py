import warnings,os,sys
import argparse
from rdkit import Chem

sys.path.append('..')
# from utils import read_pickle,split_train_valid,DrugDataset1,DrugDataset2,DrugDataLoader,CustomData
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




        self.rmodule = nn.Embedding(1316, hidden_dim)#86 963

        self.sigmoid = nn.Sigmoid()



        #####################

        if args.dataset == 'drugbank':
            self.lin = nn.Sequential(
                nn.Linear(hidden_dim * 4, hidden_dim * 8),
                # nn.BatchNorm1d(hidden_dim*8),##这行去掉 for drugbank
                nn.PReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim * 8, hidden_dim * 4),
                # nn.BatchNorm1d(hidden_dim*4),##这行去掉 for drugbank
                nn.PReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim * 4, hidden_dim),
            )
        elif args.dataset == 'twosides':
            self.lin = nn.Sequential(
                nn.Linear(hidden_dim * 4, hidden_dim * 8),
                nn.BatchNorm1d(hidden_dim*8),##这行去掉 for drugbank
                nn.PReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim * 8, hidden_dim * 4),
                nn.BatchNorm1d(hidden_dim*4),##这行去掉 for drugbank
                nn.PReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim * 4, hidden_dim),
            )
    def forward(self, triples):
        h_data, t_data, rels = triples



        x_node_h,x_line_h,h_g_node_list,h_g_line_list = self.drug_encoder(h_data)#子结构提取模块 + 子注意力模块的输出应该是原子尺度
        x_node_t, x_line_t,t_g_node_list,t_g_line_list = self.drug_encoder(t_data)


        ##################

        x_node_h,x_node_t,x_line_h,x_line_t = self.hma(x_node_h,x_node_t,x_line_h,x_line_t,h_data, t_data)

        #ablation for Fusion module
        # x_node_h,x_node_t,x_line_h,x_line_t = global_add_pool(x_node_h,h_data.batch),global_add_pool(x_node_t,t_data.batch),global_add_pool(x_line_h,h_data.edge_index_batch),global_add_pool(x_line_t,t_data.edge_index_batch)

        #

        #################################################

        rep = torch.cat([x_node_h,x_line_h, x_node_t,x_line_t], dim=-1)


        rfeat = self.rmodule(rels)

        logit = (self.lin(rep) * rfeat).sum(-1)# 97.5
        #####################################################




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
    # In case of "TypeError: __inc__() takes 3 positional arguments but 4 were given"
    # Replace with "def __inc__(self, key, value, *args, **kwargs)"
        if key == 'line_graph_edge_index':
            return self.edge_index.size(1) if self.edge_index.nelement()!=0 else 0
        return super().__inc__(key, value, *args, **kwargs)
        # In case of "TypeError: __inc__() takes 3 positional arguments but 4 were given"
        # Replace with "return super().__inc__(self, key, value, args, kwargs)"



def one_of_k_encoding(k, possible_values):
    '''
    Convert integer to one-hot representation.
    '''
    if k not in possible_values:
        raise ValueError(f"{k} is not a valid value in {possible_values}")
    return [k == e for e in possible_values]


def one_of_k_encoding_unk(x, allowable_set):
    '''
    Convert integer to one-hot representation.
    '''
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom, atom_symbols, explicit_H=True, use_chirality=False):
    '''
    Get atom features. Note that atom.GetFormalCharge() can return -1
    '''
    results = one_of_k_encoding_unk(atom.GetSymbol(), atom_symbols + ['Unknown']) + \
              one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + \
              one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), [
                  Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                  Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                    SP3D, Chem.rdchem.HybridizationType.SP3D2
              ]) + [atom.GetIsAromatic()]
    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                  [0, 1, 2, 3, 4])
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                atom.GetProp('_CIPCode'),
                ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False
                                 ] + [atom.HasProp('_ChiralityPossible')]

    results = np.array(results).astype(np.float32)

    return torch.from_numpy(results)

def edge_features(bond):
    '''
    Get bond features
    '''
    bond_type = bond.GetBondType()
    return torch.tensor([
        bond_type == Chem.rdchem.BondType.SINGLE,
        bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE,
        bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()]).long()

def generate_drug_data(id, mol_graph, atom_symbols):
    # ECFP4_FP = torch.Tensor([int(i) for i in AllChem.GetMorganFingerprintAsBitVect(mol_graph, radius=2, nBits=512).ToBitString()]).unsqueeze(0)
    # MACCS_FP = torch.Tensor([int(i) for i in MACCSkeys.GenMACCSKeys(mol_graph).ToBitString()]).unsqueeze(0)

    # (bond_i, bond_j, bond_features)
    edge_list = torch.LongTensor(
        [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), *edge_features(b)) for b in mol_graph.GetBonds()])
    # Separate (bond_i, bond_j, bond_features) to (bond_i, bond_j) and bond_features
    edge_list, edge_feats = (edge_list[:, :2], edge_list[:, 2:].float()) if len(edge_list) else (
    torch.LongTensor([]), torch.FloatTensor([]))
    # Convert the graph to undirect graph, e.g., [(1, 0)] to [(1, 0), (0, 1)]
    edge_list = torch.cat([edge_list, edge_list[:, [1, 0]]], dim=0) if len(edge_list) else edge_list  # 无向图边索引（=adj）
    edge_feats = torch.cat([edge_feats] * 2, dim=0) if len(edge_feats) else edge_feats  # 无向图边特征矩阵

    features = [(atom.GetIdx(), atom_features(atom, atom_symbols)) for atom in mol_graph.GetAtoms()]
    features.sort()
    _, features = zip(*features)
    features = torch.stack(features)  # 节点特征矩阵

    # This is the most essential step to convert a node graph to a line graph#！！！！
    line_graph_edge_index = torch.LongTensor([])
    if edge_list.nelement() != 0:
        conn = (edge_list[:, 1].unsqueeze(1) == edge_list[:, 0].unsqueeze(0)) & (
                    edge_list[:, 0].unsqueeze(1) != edge_list[:, 1].unsqueeze(0))  # 线图的构建方式
        line_graph_edge_index = conn.nonzero(as_tuple=False).T

    new_edge_index = edge_list.T
    # features节点特征矩阵   new_edge_index节点邻接矩阵    edge_feats 边特征矩阵   line_graph_edge_index线图邻接矩阵
    data = CustomData(x=features, edge_index=new_edge_index, line_graph_edge_index=line_graph_edge_index,
                      edge_attr=edge_feats, id=id)

    return data


class DrugDataset2(Dataset):
    def __init__(self, data_df, drug_graph,args):
        self.data_df = data_df
        self.drug_graph = drug_graph
        self.args = args

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        return self.data_df.iloc[index]

    # def _create_b_graph(self,edge_index,x_h, x_t):
    #     return BipartiteData(edge_index,x_h,x_t)

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

    # pred_list = []
    # h_id = []
    # t_id = []
    # rel_list = []
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
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
                print()

                print(rel_dict[str(rel_list[0])],'\t','interaction pred:', pred_list[0])
            elif args.dataset == 'DeepDDI':
                print()

                print('interaction pred:',pred_list[0])




def get_rel_dict(dataset):
    rel_dict = {}
    with open('./data/preprocessed/case4/'+dataset+'_rel_dict.txt') as f:
        for line in f:
            Y = line.strip().split('\t')[0]
            content = line.strip().split('\t')[1]
            rel_dict[Y] = content
    return rel_dict


def test(args):

    batch_size = args.batch_size

    mol1 = Chem.MolFromSmiles(args.drug1.strip())
    mol2 = Chem.MolFromSmiles(args.drug2.strip())
    if mol1==None or mol2==None:

        print('The SMILES of drugs are invalid')
        exit()

    if args.dataset == 'drugbank':
        drug_graph1 = generate_drug_data('drug1', mol1,
                                        ['Xe', 'N', 'In', 'Mn', 'K', 'Tc', 'Li', 'S', 'P', 'Si', 'Ra', 'Rb', 'Y', 'O',
                                         'Sm', 'Fe', 'B', 'Cu', 'Ba', 'Al', 'Ag', 'Hg', 'V', 'Ne', 'Ni', 'Zn', 'La',
                                         'Lu', 'Cr', 'Gd', 'Cl', 'Pt', 'Sn', 'Ga', 'Co', 'He', 'Zr', 'Ti', 'As', 'Br',
                                         'H', 'Mg', 'Sr', 'Pd', 'C', 'I', 'Ca', 'Bi', 'Sb', 'Tl', 'Au', 'F', 'Se', 'Kr',
                                         'Na', 'Mo'])

        drug_graph2 = generate_drug_data('drug2', mol2,
                                        ['Xe', 'N', 'In', 'Mn', 'K', 'Tc', 'Li', 'S', 'P', 'Si', 'Ra', 'Rb', 'Y', 'O',
                                         'Sm', 'Fe', 'B', 'Cu', 'Ba', 'Al', 'Ag', 'Hg', 'V', 'Ne', 'Ni', 'Zn', 'La',
                                         'Lu', 'Cr', 'Gd', 'Cl', 'Pt', 'Sn', 'Ga', 'Co', 'He', 'Zr', 'Ti', 'As', 'Br',
                                         'H', 'Mg', 'Sr', 'Pd', 'C', 'I', 'Ca', 'Bi', 'Sb', 'Tl', 'Au', 'F', 'Se', 'Kr',
                                         'Na', 'Mo'])
    elif args.dataset == 'twosides':
        drug_graph1 = generate_drug_data('drug1', mol1,
                                        ['Na', 'P', 'Rb', 'Mo', 'He', 'Sm', 'As', 'F', 'Br', 'Sb', 'Cu', 'Pt', 'Ne',
                                         'Mg', 'Se', 'O', 'Ra', 'Hg', 'Mn', 'Co',
                                         'Zn', 'Tl', 'Ti', 'Cl', 'K', 'Pd', 'Ga', 'Si', 'In', 'La', 'Ni', 'Fe', 'Lu',
                                         'H', 'Ag', 'Gd', 'Al', 'C', 'S', 'Bi', 'I',
                                         'N', 'Ca', 'Sn', 'Sr', 'Xe', 'Kr', 'Tc', 'V', 'Y', 'Zr', 'B', 'Au', 'Ba', 'Li',
                                         'Cr'])

        drug_graph2 = generate_drug_data('drug2', mol2,
                                        ['Na', 'P', 'Rb', 'Mo', 'He', 'Sm', 'As', 'F', 'Br', 'Sb', 'Cu', 'Pt', 'Ne',
                                         'Mg', 'Se', 'O', 'Ra', 'Hg', 'Mn', 'Co',
                                         'Zn', 'Tl', 'Ti', 'Cl', 'K', 'Pd', 'Ga', 'Si', 'In', 'La', 'Ni', 'Fe', 'Lu',
                                         'H', 'Ag', 'Gd', 'Al', 'C', 'S', 'Bi', 'I',
                                         'N', 'Ca', 'Sn', 'Sr', 'Xe', 'Kr', 'Tc', 'V', 'Y', 'Zr', 'B', 'Au', 'Ba', 'Li',
                                         'Cr'])
    elif args.dataset == 'DeepDDI':
        drug_graph1 = generate_drug_data('drug1', mol1,
                                         ['Cr', 'Ni', 'As', 'Ne', 'La', 'Ag', 'Zn', 'B', 'Pt', 'Sr', 'Gd', 'Fe', 'Zr',
                                          'Li', 'Al', 'Si', 'Se', 'Sn', 'Ga', 'Mo', 'Lu', 'Cu', 'Rb', 'Tl', 'F', 'Ra',
                                          'S', 'C', 'Au', 'Hg', 'V', 'Br', 'Ti', 'Cl', 'He', 'P', 'Mg', 'H', 'Sm', 'In',
                                          'Co', 'Sb', 'I', 'Ba', 'Y', 'K', 'Mn', 'O', 'Bi', 'Kr', 'Tc', 'Pd', 'N', 'Na',
                                          'Ca', 'Xe'])

        drug_graph2 = generate_drug_data('drug2', mol2,
                                         ['Cr', 'Ni', 'As', 'Ne', 'La', 'Ag', 'Zn', 'B', 'Pt', 'Sr', 'Gd', 'Fe', 'Zr',
                                          'Li', 'Al', 'Si', 'Se', 'Sn', 'Ga', 'Mo', 'Lu', 'Cu', 'Rb', 'Tl', 'F', 'Ra',
                                          'S', 'C', 'Au', 'Hg', 'V', 'Br', 'Ti', 'Cl', 'He', 'P', 'Mg', 'H', 'Sm', 'In',
                                          'Co', 'Sb', 'I', 'Ba', 'Y', 'K', 'Mn', 'O', 'Bi', 'Kr', 'Tc', 'Pd', 'N', 'Na',
                                          'Ca', 'Xe'])
    else:
        print('input valid dataset')
        exit()
    drug_graph = {'drug1':drug_graph1,'drug2':drug_graph2}

    test_df = []
    rel_dict = {}
    if args.dataset == 'drugbank':

        ############################
        rel_dict = get_rel_dict(args.dataset)

        for i in range(86):
            test_df.append(['drug1','drug2',args.drug1.strip(),args.drug2.strip(),i])




        test_df = pd.DataFrame(test_df, columns=['ID1', 'ID2','X1','X2','Y'])


        test_set = DrugDataset2(test_df, drug_graph, args)
        test_loader = DrugDataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
    elif args.dataset == 'twosides':
        rel_dict = get_rel_dict(args.dataset)

        for i in range(1316):
            test_df.append(['drug1', 'drug2', args.drug1.strip(), args.drug2.strip(), i])
        test_df = pd.DataFrame(test_df, columns=['ID1', 'ID2','X1','X2','Y'])


        test_set = DrugDataset2(test_df, drug_graph, args)
        test_loader = DrugDataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
    elif args.dataset == 'DeepDDI':

        test_df.append(['drug1','drug2',args.drug1.strip(),args.drug2.strip(),0])
        test_df = pd.DataFrame(test_df, columns=['ID1','ID2','X1','X2', 'Y'])

        test_set = DrugDataset2(test_df, drug_graph, args)
        test_loader = DrugDataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
    else:
        raise Exception('Error Input for --split and --dataset')

    data = next(iter(test_loader))

    node_dim = data[0].x.size(-1)
    edge_dim = 6  # data[0].edge_attr.size(-1)

    if args.dataset == 'twosides':
        args.heads = 1


    device = torch.device('cuda:'+args.device)

    if args.dataset == 'drugbank':
        model = nnModel(node_dim, edge_dim, n_iter=args.n_iter, args=args).to(device)
    elif args.dataset == 'twosides':
        model = nnModel3(node_dim, edge_dim, n_iter=args.n_iter, args=args).to(device)
    else:
        model = nnModel2(node_dim, edge_dim, n_iter=args.n_iter, args=args).to(device)



    model_path = './data/preprocessed/case4/model/'+args.dataset+'/model.pkl'

    model.load_state_dict(torch.load(model_path, map_location=device))

    rel_list = []
    if args.dataset == 'drugbank' or args.dataset == 'twosides' or args.dataset == 'DeepDDI':
        test_nn_warm(model=model, test_loader=test_loader, device=device, args=args,rel_dict=rel_dict)



if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument('--n_iter', type=int, default=10, help='number of iterations')
    parser.add_argument('--fold', type=int, default=0, help='[0, 1, 2]')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--dataset', type=str, default='drugbank', help='drugbank, twosides,or DeepDDI')
    parser.add_argument('--split', type=str, default='warm', help='warm or cold.')
    parser.add_argument('--log', type=str, default=0, help='logging or not.')
    parser.add_argument('--device', type=str, default='0', help='cuda device.')
    parser.add_argument('--num_workers', type=int, default=1, help='num_workers.')
    parser.add_argument('--heads', type=int, default=2, help='heads.')
    parser.add_argument('--drug1', type=str, default='', help='SMILES of Drug1.')
    parser.add_argument('--drug2', type=str, default='', help='SMILES of Drug2.')

    args = parser.parse_args()
    print(args)
    test(args)

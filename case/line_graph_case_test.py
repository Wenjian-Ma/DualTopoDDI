##############
'''
条形码的case
MeTDDI中给出了58个药物的关键官能团，本模型(MMDDI数据集所有数据合并起来作为训练集)针对58个药物预测出~10000个DDI，对每个DDI条目中的药物A和药物B的节点图和线图注意力分别进行收集，之后，相同药物的节点图和线图注意力各自进行求和，再进行归一化，研究发现，注意力高的地方落在官能团的节点和边上

通过上述方法求出每个分子的节点注意力和边注意力后，将具有相同关键官能团的一批分子的官能团部分注意力分数进行拼接，与全1向量求MSE，得到特定官能团节点MSE和边MSE
'''
#############

import sys
sys.path.append('..')
import warnings
import argparse
from utils import read_pickle,  DrugDataLoader, CustomData
from model import DrugEncoder
from tqdm import tqdm
import torch
from torch_scatter import scatter
import torch.nn as nn
from metric import do_compute_metrics_MMDDI
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool,global_mean_pool,GATConv
from torch_geometric.utils import softmax
from torch_geometric.data import Batch
from torch.utils.data import Dataset
import pandas as pd
import os
from rdkit import Chem
from rdkit.Chem import Draw
from matplotlib.colors import ColorConverter
import numpy as np
from tqdm import tqdm
# from sklearn.metrics.pairwise import cosine_similarity


chemicals_dict = {}

chemicals = pd.read_csv('/media/ST-18T/Ma/HA-DDI/data/preprocessed/case/58_chemicals.csv',delimiter=',',header=None)

chemicals_substructure_dict = {}
substructure_chemicals_dict = {}
for i in range(58):
    chemicals_dict[chemicals[0].tolist()[i]] = chemicals[2].tolist()[i]

    substructure_name = chemicals[3].tolist()[i]
    if '-' in substructure_name:
        substructure_name = substructure_name.split('-')[1]

    chemicals_substructure_dict[chemicals[0].tolist()[i]] = substructure_name

for i,j in chemicals_substructure_dict.items():
    if j not in substructure_chemicals_dict.keys():
        substructure_chemicals_dict[j] = []
    substructure_chemicals_dict[j].append(i)


substructures_dict = {'Benzodioxole':'C1OC2=CC=CC=C2O1','Acetylene':'C#C','Alkylimidazole':'Cc1ncc[nH]1','Amine':'CN(C)C',#Dimethylamine
'Cyclopropylamine':'C1CC1N','Furan':'C1=COC=C1','Hydrazine':'NN','Imidazole':'C1=CN=CN1','Morpholine':'C1COCCN1','Pyridine':'C1=CC=NC=C1','Thiophene':'C1=CSC=C1','Triazole':'C1=NC=NN1'}#1,2,4-Triazole

drug_substruct_dict = {}
for i,j in chemicals_substructure_dict.items():
    drug_substruct_dict[i] = substructures_dict[j]
###########################
drug_substruct_dict['Fluoxetine'] = 'CNC'
drug_substruct_dict['Lapatinib'] = 'CNC'
drug_substruct_dict['N-desethylamiodarone'] = 'CNC'
drug_substruct_dict['Sertraline'] = 'CNC'
#drug_substruct_dict---药物名为key，子结构SMILES为value
#chemicals_dict---药物名为key，药物SMILES为value


drug_substruct_attn_dict = {}
drug_substruct_line_attn_dict = {}
for drug_name,smiles in chemicals_dict.items():
    chemicals_mol = Chem.MolFromSmiles(smiles)#药物分子Mol
    func_group_mol = Chem.MolFromSmiles(drug_substruct_dict[drug_name])#官能团Mol
    matches = chemicals_mol.GetSubstructMatches(func_group_mol)

    edge_list = torch.LongTensor([(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in chemicals_mol.GetBonds()])
    edge_list = torch.cat([edge_list, edge_list[:, [1, 0]]], dim=0).numpy() if len(edge_list) else edge_list
    # [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in func_group_mol.GetBonds()]
    line_matches = []
    if matches:
        print(drug_name,f"官能团匹配位置: {matches}")
        atoms_num = chemicals_mol.GetNumAtoms()
        real_attn = np.zeros((atoms_num,1))
        real_line_attn = np.zeros((edge_list.shape[0],1))
        if len(matches) == 1:
            matches = np.array(matches[0])
        else:
            matches = np.array(sum(list(matches),()))
        real_attn[matches] = 1
        drug_substruct_attn_dict[drug_name] = real_attn.flatten()

        for idx,(begin_id,end_id) in enumerate(edge_list):
            if begin_id in list(matches) and end_id in list(matches):
                line_matches.append(idx)
        line_matches = np.array(line_matches)
        real_line_attn[line_matches] = 1
        drug_substruct_line_attn_dict[drug_name] = real_line_attn.flatten()[:int(real_line_attn.shape[0]/2)]

    else:
        print(drug_name,"没有找到匹配的官能团")

#drug_substruct_attn_dict 药物名为key，value为一个向量，其中，关键官能团所占元素为1，其它部分元素为0


def del_neg_samples(pred_label,splited_p_t_list,splited_p_t_line_list,splited_line_to_node_attn_list,drug_name_list):
    index = np.sort(np.nonzero(pred_label)[0])[::-1]
    for i in index:
        splited_p_t_list.pop(i)
        splited_p_t_line_list.pop(i)
        splited_line_to_node_attn_list.pop(i)
        drug_name_list.pop(i)
    return splited_p_t_list,splited_p_t_line_list,splited_line_to_node_attn_list,drug_name_list

def normalized_min_max(array):
    arr_min = array.min()
    arr_max = array.max()

    if arr_min == arr_max:
        arr_normalized = np.ones_like(array)/2
    else:
        arr_normalized = (array - arr_min) / (arr_max - arr_min)

    return arr_normalized
def cal_attn(splited_p_t_list,splited_p_t_line_list,splited_line_to_node_attn_list,drug_name_list,drug_substruct_attn_dict):
    drug_predicted_node_attn_dict = {}
    drug_predicted_line2node_attn_dict = {}
    drug_predicted_line_attn_dict = {}

    for i in range(len(splited_p_t_list)):
        drug_name = drug_name_list[i]
        node_attn = splited_p_t_list[i].cpu().numpy().flatten()

        line_attn = splited_p_t_line_list[i].cpu().numpy().flatten()
        line_attn = (line_attn[:int(line_attn.size/2)] + line_attn[int(line_attn.size/2):])/2


        line2node_attn = splited_line_to_node_attn_list[i].cpu().numpy().flatten()

        real_attn = drug_substruct_attn_dict[drug_name]

        if drug_name not in drug_predicted_node_attn_dict.keys():
            drug_predicted_node_attn_dict[drug_name] = []
            drug_predicted_line_attn_dict[drug_name] = []
            drug_predicted_line2node_attn_dict[drug_name] = []
        drug_predicted_node_attn_dict[drug_name].append(node_attn)
        drug_predicted_line_attn_dict[drug_name].append(line_attn)
        drug_predicted_line2node_attn_dict[drug_name].append(line2node_attn)
    for i in drug_predicted_node_attn_dict.keys():
        a = sum(drug_predicted_node_attn_dict[i])
        b = sum(drug_predicted_line2node_attn_dict[i])
        c = sum(drug_predicted_line_attn_dict[i])

        drug_predicted_node_attn_dict[i] = normalized_min_max(a)
        drug_predicted_line2node_attn_dict[i] = normalized_min_max(b)
        drug_predicted_line_attn_dict[i] = normalized_min_max(c)
    node_mse_dict = {}
    line_mse_dict = {}
    line2node_mse_dict = {}

    local_node_mse_dict = {}
    local_line_mse_dict = {}
    local_line2node_mse_dict = {}


    for i in drug_predicted_node_attn_dict.keys():
        node_mse = np.sum((drug_substruct_attn_dict[i]-drug_predicted_node_attn_dict[i])**2)/len(drug_substruct_attn_dict[i])
        line2node_mse = np.sum((drug_substruct_attn_dict[i] - drug_predicted_line2node_attn_dict[i])**2)/len(drug_substruct_attn_dict[i])
        line_mse = np.sum((drug_substruct_line_attn_dict[i]-drug_predicted_line_attn_dict[i])**2)/len(drug_substruct_line_attn_dict[i])

        node_mse_dict[i] = node_mse
        line_mse_dict[i] = line_mse
        line2node_mse_dict[i] = line2node_mse

        a = np.where(drug_substruct_attn_dict[i]==1)[0]
        a_line = np.where(drug_substruct_line_attn_dict[i]==1)[0]
        local_node_mse = np.sum((drug_substruct_attn_dict[i][a] - drug_predicted_node_attn_dict[i][a]) ** 2) / len(
            drug_substruct_attn_dict[i][a])
        local_line_mse = np.sum((drug_substruct_line_attn_dict[i][a_line] - drug_predicted_line_attn_dict[i][a_line]) ** 2) / len(
            drug_substruct_line_attn_dict[i][a_line])
        local_line2node_mse = np.sum((drug_substruct_attn_dict[i][a] - drug_predicted_line2node_attn_dict[i][a]) ** 2) / len(
            drug_substruct_attn_dict[i][a])

        local_node_mse_dict[i] = local_node_mse
        local_line_mse_dict[i] = local_line_mse
        local_line2node_mse_dict[i] = local_line2node_mse

    func_group_mse = {}

    for func_group,chemicals_name_list in substructure_chemicals_dict.items():
        node_attn_for_func_group = []
        line_attn_for_func_group = []
        for chemicals_name in chemicals_name_list:
            a = np.where(drug_substruct_attn_dict[chemicals_name]==1)[0]
            a_line = np.where(drug_substruct_line_attn_dict[chemicals_name]==1)[0]
            if chemicals_name not in drug_predicted_node_attn_dict.keys() or chemicals_name not in drug_predicted_line_attn_dict.keys():
                continue
            node_attn_for_func_group.append(np.sum((drug_predicted_node_attn_dict[chemicals_name][a]-np.ones_like(drug_predicted_node_attn_dict[chemicals_name][a]))**2)/len(drug_predicted_node_attn_dict[chemicals_name][a]))
            line_attn_for_func_group.append(np.sum((drug_predicted_line_attn_dict[chemicals_name][a_line]-np.ones_like(drug_predicted_line_attn_dict[chemicals_name][a_line]))**2)/len(drug_predicted_line_attn_dict[chemicals_name][a_line]))
        mean_node_attn_for_func_group = np.mean(node_attn_for_func_group)
        mean_line_attn_for_func_group = np.mean(line_attn_for_func_group)
        median_node_attn_for_func_group = np.median(node_attn_for_func_group)
        median_line_attn_for_func_group = np.median(line_attn_for_func_group)

        # ground_truth_func_node = np.ones_like(node_attn_for_func_group)
        # ground_truth_func_line = np.ones_like(line_attn_for_func_group)
        #
        # node_func_mse = np.sum((ground_truth_func_node - node_attn_for_func_group) ** 2) / len(
        #     ground_truth_func_node)
        # line_func_mse = np.sum((ground_truth_func_line - line_attn_for_func_group) ** 2) / len(
        #     ground_truth_func_line)
        func_group_mse[func_group] = [mean_node_attn_for_func_group,mean_line_attn_for_func_group,median_node_attn_for_func_group,median_line_attn_for_func_group]


    mean_node_mse = np.mean(np.array(list(node_mse_dict.values())))
    std_node_mse = np.std(np.array(list(node_mse_dict.values())))
    mean_line_mse = np.mean(np.array(list(line_mse_dict.values())))
    std_line_mse = np.std(np.array(list(line_mse_dict.values())))
    mean_line2node_mse = np.mean(np.array(list(line2node_mse_dict.values())))
    std_line2node_mse = np.std(np.array(list(line2node_mse_dict.values())))


    local_mean_node_mse = np.mean(np.array(list(local_node_mse_dict.values())))
    local_std_node_mse = np.std(np.array(list(local_node_mse_dict.values())))
    local_mean_line_mse = np.mean(np.array(list(local_line_mse_dict.values())))
    local_std_line_mse = np.std(np.array(list(local_line_mse_dict.values())))
    local_mean_line2node_mse = np.mean(np.array(list(local_line2node_mse_dict.values())))
    local_std_line2node_mse = np.std(np.array(list(local_line2node_mse_dict.values())))

    ################

    sorted_mean_node_mse = sorted(node_mse_dict.items(), key=lambda x: x[1], reverse=True)

    sorted_mean_node_mse = dict(sorted_mean_node_mse)

    sorted_local_mean_node_mse = sorted(local_node_mse_dict.items(), key=lambda x: x[1], reverse=True)

    sorted_local_mean_node_mse = dict(sorted_local_mean_node_mse)

    ################

    sorted_mean_line_mse = sorted(line_mse_dict.items(), key=lambda x: x[1], reverse=True)

    sorted_mean_line_mse = dict(sorted_mean_line_mse)

    sorted_local_mean_line_mse = sorted(local_line_mse_dict.items(), key=lambda x: x[1], reverse=True)

    sorted_local_mean_line_mse = dict(sorted_local_mean_line_mse)

    ################
    #func_group_mse  key为官能团名字，value为列表，第一个值为该官能团对应的原子MSE，第二个值为对应的化学键MSE
    func_group_mse

    return mean_node_mse,std_node_mse,mean_line2node_mse,std_line2node_mse,mean_line_mse,std_line_mse,local_mean_node_mse,local_std_node_mse,local_mean_line2node_mse,local_std_line2node_mse,local_mean_line_mse,local_std_line_mse,drug_predicted_node_attn_dict,drug_predicted_line2node_attn_dict,drug_predicted_line_attn_dict




class DrugDataset3(Dataset):
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

            try:
                if (h_graph.x.shape[0]==1 or h_graph.edge_index.shape[0]==0 or h_graph.edge_attr.shape[0]==0 or h_graph.line_graph_edge_index.shape[0]==0) or (t_graph.x.shape[0]==1 or t_graph.edge_index.shape[0]==0 or t_graph.edge_attr.shape[0]==0 or t_graph.line_graph_edge_index.shape[0]==0) :
                    continue
            except Exception as e:
                print()

            head_list.append(h_graph)
            tail_list.append(t_graph)
            label_list.append(torch.LongTensor([Y]))

        head_pairs = Batch.from_data_list(head_list, follow_batch=['edge_index','line_graph_edge_index'])
        tail_pairs = Batch.from_data_list(tail_list, follow_batch=['edge_index','line_graph_edge_index'])
        # label = F.one_hot(torch.cat(label_list, dim=0),num_classes=4)
        label = torch.cat(label_list, dim=0)
        return head_pairs, tail_pairs, label


class hierarchical_mutual_attn(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.Wh_transform = nn.Linear(hidden_dim, hidden_dim)
        self.Wt_transform = nn.Linear(hidden_dim, hidden_dim)

        self.Wh_transform_line = nn.Linear(hidden_dim, hidden_dim)#hidden_dim
        self.Wt_transform_line = nn.Linear(hidden_dim, hidden_dim)

        # w = nn.Parameter(torch.Tensor(hidden_dim, 1))
        # self.dim = hidden_dim
        # self.w = nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))

    def forward(self, x_node_h,x_node_t,x_line_h,x_line_t,h_data, t_data):

        x_h1 = self.Wh_transform(x_node_h)
        x_t1 = self.Wt_transform(x_node_t)

        x_t2 = self.Wh_transform(x_node_t)
        x_h2 = self.Wt_transform(x_node_h)
        ##################################
        # m1 = x_h.size()[0]
        # m2 = x_t.size()[0]
        # c1 = x_h.repeat(1, m2).view(m1, m2)
        # c2 = x_t.repeat(m1, 1).view(m1, m2)
        # alpha = torch.tanh(c1 * c2)
        ###################################
        # alpha = torch.tanh(torch.mm(x_h1, torch.t(x_t1)))
        alpha = torch.tanh(torch.mm(x_h1, torch.t(x_t1))+torch.t(torch.mm(x_t2,torch.t(x_h2))))

        b_t = torch.diag(global_mean_pool(alpha,h_data.batch)[t_data.batch])
        p_t = softmax(b_t,t_data.batch).view(-1,1)
        s_t = global_add_pool(p_t * x_node_t,t_data.batch)#+x_node_t

        b_h = torch.diag(global_mean_pool(torch.t(alpha),t_data.batch)[h_data.batch])
        p_h = softmax(b_h,h_data.batch).view(-1,1)
        s_h = global_add_pool(p_h * x_node_h,h_data.batch)#+x_node_h

        x_h1_line = self.Wh_transform_line(x_line_h)
        x_t1_line = self.Wt_transform_line(x_line_t)

        x_t2_line = self.Wh_transform_line(x_line_t)
        x_h2_line = self.Wt_transform_line(x_line_h)
        ##################################
        # m1_line = x_h_line.size()[0]
        # m2_line = x_t_line.size()[0]
        # c1_line = x_h_line.repeat(1, m2_line).view(m1_line, m2_line)
        # c2_line = x_t_line.repeat(m1_line, 1).view(m1_line, m2_line)
        # alpha_line = torch.tanh(c1_line * c2_line)
        ###################################
        # alpha_line = torch.tanh(torch.mm(x_h1_line, torch.t(x_t1_line)))
        alpha_line = torch.tanh(torch.mm(x_h1_line, torch.t(x_t1_line))+torch.t(torch.mm(x_t2_line,torch.t(x_h2_line))))

        b_t_line = torch.diag(global_mean_pool(alpha_line, h_data.edge_index_batch)[t_data.edge_index_batch])
        p_t_line = softmax(b_t_line, t_data.edge_index_batch).view(-1, 1)
        s_t_line = global_add_pool(p_t_line * x_line_t, t_data.edge_index_batch)#+x_line_t

        b_h_line = torch.diag(global_mean_pool(torch.t(alpha_line), t_data.edge_index_batch)[h_data.edge_index_batch])
        p_h_line = softmax(b_h_line, h_data.edge_index_batch).view(-1, 1)
        s_h_line = global_add_pool(p_h_line * x_line_h, h_data.edge_index_batch)#+x_line_h




        return s_h,s_t,s_h_line,s_t_line,p_h,p_t,p_h_line,p_t_line


class nnModel_MMDDI(torch.nn.Module):
    def __init__(self, in_dim, edge_in_dim, hidden_dim=64, n_iter=10,args=None):
        super(nnModel_MMDDI, self).__init__()

        self.drug_encoder = DrugEncoder(in_dim, edge_in_dim, hidden_dim, n_iter=n_iter,args=args)

        self.hma = hierarchical_mutual_attn(hidden_dim)

        #####################

        if args.dataset == 'drugbank' or args.dataset == 'ChChMiner' or args.dataset == 'MMDDI':
            self.lin = nn.Sequential(
                nn.Linear(hidden_dim * 4, hidden_dim * 8),
                # nn.BatchNorm1d(hidden_dim*8),##这行去掉 for drugbank
                nn.PReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim * 8, hidden_dim * 4),
                # nn.BatchNorm1d(hidden_dim*4),##这行去掉 for drugbank
                nn.PReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim * 4, 4),
            )
        elif args.dataset == 'twosides' or args.dataset == 'ZhangDDI' or args.dataset == 'DeepDDI':
            self.lin = nn.Sequential(
                nn.Linear(hidden_dim * 4, hidden_dim * 8),
                nn.BatchNorm1d(hidden_dim*8),##这行去掉 for drugbank
                nn.PReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim * 8, hidden_dim * 4),
                nn.BatchNorm1d(hidden_dim*4),##这行去掉 for drugbank
                nn.PReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim * 4, 4),
            )
    def forward(self, triples):
        h_data, t_data = triples



        x_node_h,x_line_h,h_g_node_list,h_g_line_list = self.drug_encoder(h_data)#子结构提取模块 + 子注意力模块的输出应该是原子尺度
        x_node_t, x_line_t,t_g_node_list,t_g_line_list = self.drug_encoder(t_data)


        ##################

        x_node_h,x_node_t,x_line_h,x_line_t,p_h,p_t,p_h_line,p_t_line = self.hma(x_node_h,x_node_t,x_line_h,x_line_t,h_data, t_data)

        mark = list(torch.unique(t_data.batch, return_counts=True)[1].cpu().tolist())
        mark_line = list(torch.unique(t_data.edge_index_batch, return_counts=True)[1].cpu().tolist())
        splited_p_t = torch.split(p_t, mark, dim=0)#单分子中（节点图），每个原子的注意力
        splited_p_t_line = torch.split(p_t_line, mark_line, dim=0)#单分子中（线图），每个化学键的注意力

        line_to_node_attn = scatter(p_t_line, t_data.edge_index[1], dim_size=t_data.x.size(0), dim=0, reduce='add')
        splited_line_to_node_attn = torch.split(line_to_node_attn, mark, dim=0)
        #################################################

        rep = torch.cat([x_node_h,x_line_h, x_node_t,x_line_t], dim=-1)



        logit = self.lin(rep)
        #####################################################




        output = logit

        return output,h_g_node_list,h_g_line_list,t_g_node_list,t_g_line_list,splited_p_t,splited_p_t_line,splited_line_to_node_attn

def test_nn(model, external_loader, device, args):
    criterion = nn.CrossEntropyLoss()


    #########################
    #########################
    model.eval()
    pred_list = []
    label_list = []
    splited_p_t_list = []
    splited_p_t_line_list = []
    splited_line_to_node_attn_list = []
    total_loss_test = 0

    with torch.no_grad():

        for idx, data in enumerate(
                tqdm(external_loader, mininterval=0.5, desc='Evaluating', leave=False, ncols=50)):
            head_pairs, tail_pairs, label = [d.to(device) for d in data]
            pred, _, _, _, _,splited_p_t,splited_p_t_line,splited_line_to_node_attn = model((head_pairs, tail_pairs))
            loss = criterion(pred, label)

            pred = F.softmax(pred, dim=-1)

            # pred_cls = torch.sigmoid(pred)
            pred_list.append(pred.detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())

            splited_p_t_list.extend(splited_p_t)
            splited_p_t_line_list.extend(splited_p_t_line)
            splited_line_to_node_attn_list.extend(splited_line_to_node_attn)


            total_loss_test = total_loss_test + loss

    pred_probs = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)
    total_loss_test = total_loss_test / (idx + 1)

    pred_label = np.argmax(pred_probs, axis=-1)

    drug_name_list = []
    with open('/media/ST-18T/Ma/HA-DDI/data/preprocessed/case/ddi_for_58.csv') as f:
        for idx,line in enumerate(f):
            if idx==0:
                continue
            drug_name_list.append(line.strip().split(',')[5])




    splited_p_t_list,splited_p_t_line_list,splited_line_to_node_attn_list,drug_name_list = del_neg_samples(pred_label,splited_p_t_list,splited_p_t_line_list,splited_line_to_node_attn_list,drug_name_list)

    mean_node_mse,std_node_mse,mean_line2node_mse,std_line2node_mse,mean_line_mse,std_line_mse,local_mean_node_mse,local_std_node_mse,local_mean_line2node_mse,local_std_line2node_mse,local_mean_line_mse,local_std_line_mse,drug_predicted_node_attn_dict,drug_predicted_line2node_attn_dict,drug_predicted_line_attn_dict = cal_attn(splited_p_t_list,splited_p_t_line_list,splited_line_to_node_attn_list,drug_name_list,drug_substruct_attn_dict)

    for i in drug_predicted_node_attn_dict.keys():
        a1 = drug_substruct_attn_dict[i]
        a2 = drug_predicted_node_attn_dict[i]
        a3 = drug_predicted_line2node_attn_dict[i]

        a4 = drug_substruct_line_attn_dict[i]
        a5 = drug_predicted_line_attn_dict[i]


        # with open('/media/ST-18T/Ma/HA-DDI/data/preprocessed/case/attn_data_fig/node_csv_files/'+i+'.csv','a') as f:
        #     for j in range(len(a1)):
        #         f.write(i+','+str(a1[j]) + ',' + str(a2[j]) + '\n')
        #
        # with open('/media/ST-18T/Ma/HA-DDI/data/preprocessed/case/attn_data_fig/line_csv_files/'+i+'.csv','a') as f:
        #     for j in range(len(a4)):
        #         f.write(i+','+str(a4[j]) + ',' + str(a5[j]) + '\n')



    acc2, auroc2, f1_score2, precision2, recall2, ap2 = do_compute_metrics_MMDDI(pred_probs, label)

    msg2 = " Test_loss-%.4f, Test_acc-%.4f, Test_auroc-%.4f, Test_f1_score-%.4f, Test_prec-%.4f, Test_rec-%.4f, Test_ap-%.4f" % (
        total_loss_test, acc2, auroc2, f1_score2, precision2, recall2, ap2)

    print(msg2)

    print(
        '节点图全局MSE为{:.3f}/{:.3f}\t线图全局MSE为{:.3f}/{:.3f}\t边全局MSE为{:.3f}/{:.3f}\t节点图官能团局部MSE为{:.3f}/{:.3f}\t线图官能团局部MSE为{:.3f}/{:.3f}\t边局部MSE为{:.3f}/{:.3f}'.format(
            mean_node_mse, std_node_mse, mean_line2node_mse, std_line2node_mse,mean_line_mse,std_line_mse, local_mean_node_mse, local_std_node_mse,
            local_mean_line2node_mse, local_std_line2node_mse,local_mean_line_mse, local_std_line_mse))


def test(args):
    root = '/media/ST-18T/Ma/HA-DDI/data/preprocessed/' + args.dataset
    batch_size = args.batch_size

    drug_graph = read_pickle(os.path.join('/media/ST-18T/Ma/HA-DDI/data/preprocessed/case', 'drug_data.pkl'))

    external_df = pd.read_csv(os.path.join('/media/ST-18T/Ma/HA-DDI/data/preprocessed/case', f'ddi_for_58.csv'), delimiter=',')

    external_set = DrugDataset3(external_df, drug_graph, args)

    external_loader = DrugDataLoader(external_set, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)


    data = next(iter(external_loader))

    node_dim = data[0].x.shape[1]
    edge_dim = 6  # data[0].edge_attr.size(-1)

    device = torch.device('cuda:' + args.device)

    model = nnModel_MMDDI(node_dim, edge_dim, n_iter=args.n_iter, args=args).to(device)

    model.load_state_dict(torch.load('/media/ST-18T/Ma/HA-DDI/data/preprocessed/case/model.pkl', map_location=device))

    test_nn(model=model,external_loader=external_loader,device=device, args=args)



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
    test(args)
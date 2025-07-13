
import torch
from collections import defaultdict,Counter
from sklearn.model_selection import StratifiedShuffleSplit
from rdkit import Chem
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle,warnings
import os
from utils import CustomData

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
            one_of_k_encoding(atom.GetDegree(),[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + \
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

def generate_drug_data(id,mol_graph, atom_symbols):
    #ECFP4_FP = torch.Tensor([int(i) for i in AllChem.GetMorganFingerprintAsBitVect(mol_graph, radius=2, nBits=512).ToBitString()]).unsqueeze(0)
    #MACCS_FP = torch.Tensor([int(i) for i in MACCSkeys.GenMACCSKeys(mol_graph).ToBitString()]).unsqueeze(0)

    # (bond_i, bond_j, bond_features)
    edge_list = torch.LongTensor([(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), *edge_features(b)) for b in mol_graph.GetBonds()])
    # Separate (bond_i, bond_j, bond_features) to (bond_i, bond_j) and bond_features
    edge_list, edge_feats = (edge_list[:, :2], edge_list[:, 2:].float()) if len(edge_list) else (torch.LongTensor([]), torch.FloatTensor([]))
    # Convert the graph to undirect graph, e.g., [(1, 0)] to [(1, 0), (0, 1)]
    edge_list = torch.cat([edge_list, edge_list[:, [1, 0]]], dim=0) if len(edge_list) else edge_list#无向图边索引（=adj）
    edge_feats = torch.cat([edge_feats]*2, dim=0) if len(edge_feats) else edge_feats#无向图边特征矩阵

    features = [(atom.GetIdx(), atom_features(atom, atom_symbols)) for atom in mol_graph.GetAtoms()]
    features.sort()
    _, features = zip(*features)
    features = torch.stack(features)#节点特征矩阵

    # This is the most essential step to convert a node graph to a line graph#！！！！
    line_graph_edge_index = torch.LongTensor([])
    if edge_list.nelement() != 0:
        conn = (edge_list[:, 1].unsqueeze(1) == edge_list[:, 0].unsqueeze(0)) & (edge_list[:, 0].unsqueeze(1) != edge_list[:, 1].unsqueeze(0))#线图的构建方式
        line_graph_edge_index = conn.nonzero(as_tuple=False).T

    new_edge_index = edge_list.T
    #features节点特征矩阵   new_edge_index节点邻接矩阵    edge_feats 边特征矩阵   line_graph_edge_index线图邻接矩阵
    data = CustomData(x=features, edge_index=new_edge_index, line_graph_edge_index=line_graph_edge_index, edge_attr=edge_feats,id = id)

    return data


def save_data(data):

    filename = '/media/ST-18T/Ma/TIDE/data/preprocessed/case/drug_data.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f'\nData saved as {filename}!')


def load_drug_mol_data():

    data1 = pd.read_csv('/media/ST-18T/Ma/TIDE/data/preprocessed/AUC_FC/fold1.csv', delimiter=',')
    data2 = pd.read_csv('/media/ST-18T/Ma/TIDE/data/preprocessed/AUC_FC/fold2.csv', delimiter=',')
    data3 = pd.read_csv('/media/ST-18T/Ma/TIDE/data/preprocessed/AUC_FC/fold3.csv', delimiter=',')
    data4 = pd.read_csv('/media/ST-18T/Ma/TIDE/data/preprocessed/AUC_FC/fold4.csv', delimiter=',')
    data5 = pd.read_csv('/media/ST-18T/Ma/TIDE/data/preprocessed/AUC_FC/fold5.csv', delimiter=',')

    data = pd.concat([data1,data2,data3,data4,data5])

    drug_id_mol_tup = []
    symbols = list()
    drug_smile_dict = {}

    for id1, id2, smiles1, smiles2, relation in zip(data['ID1'],data['ID2'],data['X1'], data['X2'],data['Y']):
        drug_smile_dict[id1] = smiles1
        drug_smile_dict[id2] = smiles2

    #######for external AUC_FC##########

    with open('/media/ST-18T/Ma/TIDE/data/preprocessed/AUC_FC/External.csv') as f:
        for id,line in enumerate(f):
            if id ==0:
                continue
            id1 = line.strip().split(',')[0]
            smiles1 = line.strip().split(',')[2]
            id2 = line.strip().split(',')[1]
            smiles2 = line.strip().split(',')[3]
            drug_smile_dict[id1] = smiles1
            drug_smile_dict[id2] = smiles2

    ####################################

    # #######for external DDInter##########
    #
    # with open('/media/ST-18T/Ma/TIDE/data/preprocessed/MMDDI/DDInter.csv') as f:
    #     for id, line in enumerate(f):
    #         if id == 0:
    #             continue
    #         id1 = line.strip().split(',')[0]
    #         smiles1 = line.strip().split(',')[2]
    #         id2 = line.strip().split(',')[1]
    #         smiles2 = line.strip().split(',')[3]
    #         drug_smile_dict[id1] = smiles1
    #         drug_smile_dict[id2] = smiles2
    # ##################################################################
    # with open('/media/ST-18T/Ma/TIDE/data/preprocessed/case/ddi_for_58.csv') as f:
    #     for id, line in enumerate(f):
    #         if id == 0:
    #             continue
    #
    #         id1 = line.strip().split(',')[4]
    #         smiles1 = line.strip().split(',')[0]
    #         id2 = line.strip().split(',')[5]
    #         smiles2 = line.strip().split(',')[1]
    #
    #         drug_smile_dict[id1] = smiles1
    #         drug_smile_dict[id2] = smiles2
    #
    # ####################################


    mol_dict = {}
    for id, smiles in drug_smile_dict.items():
        mol =  Chem.MolFromSmiles(smiles.strip())
        if mol is not None:
            drug_id_mol_tup.append((id, mol))
            symbols.extend(atom.GetSymbol() for atom in mol.GetAtoms())
            mol_dict[id] = mol
    symbols = list(set(symbols))#去重后数据集中的原子类型
    drug_data = {id: generate_drug_data(id,mol, symbols) for id, mol in tqdm(drug_id_mol_tup, desc='Processing drugs')}
    #drug_data = {id:data for id,data in drug_data.items() if data.x.shape[0]!=1 and data.edge_index.shape[0]!=0 and data.edge_attr.shape[0]!=0 and data.line_graph_edge_index.shape[0]!=0}#Removing invalid molecules on the basis of 1706 mols
    # save_data(drug_data)
    # save_data(mol_dict, 'mol_dict.pkl', args)
    return drug_data

load_drug_mol_data()

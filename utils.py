import pickle,torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch_geometric.data import Batch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
import numpy as np
import torch.nn.functional as F

def read_pickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj

def split_train_valid(data_df, fold, val_ratio=0.2):
    cv_split = StratifiedShuffleSplit(n_splits=2, test_size=val_ratio, random_state=fold)
    train_index, val_index = next(iter(cv_split.split(X=range(len(data_df)), y=data_df['Y'])))

    train_df = data_df.iloc[train_index]
    val_df = data_df.iloc[val_index]

    return train_df, val_df

# def get_bipartite_graph(x1,x2):
#     x1 = np.arange(0,x1)
#     x2 = np.arange(0,x2)
#     edge_list = torch.LongTensor(np.meshgrid(x1,x2))
#     edge_list = torch.stack([edge_list[0].reshape(-1),edge_list[1].reshape(-1)])
#     return edge_list


# class BipartiteData(Data):
#     def __init__(self, edge_index=None, x_h=None, x_t=None):
#         super().__init__()
#         self.edge_index = edge_index
#         self.x_h = x_h
#         self.x_t = x_t
#     def __inc__(self, key, value, *args, **kwargs):
#         if key == 'edge_index':
#             return torch.tensor([[self.x_h.size(0)], [self.x_t.size(0)]])
#         else:
#             return super().__inc__(key, value, *args, **kwargs)

class DrugDataset1(Dataset):
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
        rel_list = []

        # bi_list = []
        # bi_line_list = []
        for row in batch:
            Drug1_ID, Drug2_ID, Y, Neg_samples = row['Drug1_ID'], row['Drug2_ID'], row['Y'], row['Neg samples']
            if self.args.dataset == 'drugbank':
                Neg_ID, Ntype = Neg_samples.split('$')
                h_graph = self.drug_graph.get(Drug1_ID)
                t_graph = self.drug_graph.get(Drug2_ID)
                n_graph = self.drug_graph.get(Neg_ID)

                pos_pair_h = h_graph
                pos_pair_t = t_graph

                if Ntype == 'h':
                    neg_pair_h = n_graph
                    neg_pair_t = t_graph
                else:
                    neg_pair_h = h_graph
                    neg_pair_t = n_graph
            else:
                h_graph = self.drug_graph.get(Drug1_ID)
                t_graph = self.drug_graph.get(Drug2_ID)
                n_graph = self.drug_graph.get(Neg_samples)

                pos_pair_h = h_graph
                pos_pair_t = t_graph
                neg_pair_h = n_graph
                neg_pair_t = t_graph

            if (pos_pair_h.x.shape[0]==1 or pos_pair_h.edge_index.shape[0]==0 or pos_pair_h.edge_attr.shape[0]==0 or pos_pair_h.line_graph_edge_index.shape[0]==0) or (pos_pair_t.x.shape[0]==1 or pos_pair_t.edge_index.shape[0]==0 or pos_pair_t.edge_attr.shape[0]==0 or pos_pair_t.line_graph_edge_index.shape[0]==0) or (neg_pair_h.x.shape[0]==1 or neg_pair_h.edge_index.shape[0]==0 or neg_pair_h.edge_attr.shape[0]==0 or neg_pair_h.line_graph_edge_index.shape[0]==0) or (neg_pair_t.x.shape[0]==1 or neg_pair_t.edge_index.shape[0]==0 or neg_pair_t.edge_attr.shape[0]==0 or neg_pair_t.line_graph_edge_index.shape[0]==0):
                continue

            # self.mol_dict[Drug1_ID]
            # self.mol_dict[Drug2_ID]
            # self.mol_dict[Neg_samples]
            # self.mol_dict[Drug2_ID]

            # bi_list.append(self._create_b_graph(get_bipartite_graph(pos_pair_h.x.shape[0],pos_pair_t.x.shape[0]),pos_pair_h.x,pos_pair_t.x))
            # bi_list.append(self._create_b_graph(get_bipartite_graph(neg_pair_h.x.shape[0],neg_pair_t.x.shape[0]),neg_pair_h.x,neg_pair_t.x))

            # bi_line_list.append(self._create_b_graph(get_bipartite_graph(pos_pair_h.edge_attr.shape[0],pos_pair_t.edge_attr.shape[0]),pos_pair_h.edge_attr,pos_pair_t.edge_attr))
            # bi_line_list.append(self._create_b_graph(get_bipartite_graph(neg_pair_h.edge_attr.shape[0], neg_pair_t.edge_attr.shape[0]), neg_pair_h.edge_attr,neg_pair_t.edge_attr))


            head_list.append(pos_pair_h)
            head_list.append(neg_pair_h)

            tail_list.append(pos_pair_t)
            tail_list.append(neg_pair_t)

            rel_list.append(torch.LongTensor([Y]))
            rel_list.append(torch.LongTensor([Y]))

            label_list.append(torch.FloatTensor([1]))
            label_list.append(torch.FloatTensor([0]))

        head_pairs = Batch.from_data_list(head_list, follow_batch=['edge_index','line_graph_edge_index'])
        tail_pairs = Batch.from_data_list(tail_list, follow_batch=['edge_index','line_graph_edge_index'])
        rel = torch.cat(rel_list, dim=0)
        label = torch.cat(label_list, dim=0)

        # if self.data_df.shape[0] == 2928823:
        #     with open('/media/ST-18T/Ma/MR-GNN-master/datasets/twosides_train.txt','a') as f:#122756   38362
        #         for i in range(len(head_list)):
        #             f.write(head_list[i].id+'\t'+tail_list[i].id+'\t'+str(rel_list[i].item())+'\t'+str(label_list[i].item())+'\n')
        # elif self.data_df.shape[0] == 915258:
        #     with open('/media/ST-18T/Ma/MR-GNN-master/datasets/twosides_test.txt','a') as f:#122756   38362
        #         for i in range(len(head_list)):
        #             f.write(head_list[i].id+'\t'+tail_list[i].id+'\t'+str(rel_list[i].item())+'\t'+str(label_list[i].item())+'\n')
        # b_pairs = Batch.from_data_list(bi_list)
        # b_line_pairs = Batch.from_data_list(bi_line_list)

        return head_pairs, tail_pairs, rel, label#,b_line_pairs

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
            label_list.append(torch.FloatTensor([Y]))

        head_pairs = Batch.from_data_list(head_list, follow_batch=['edge_index','line_graph_edge_index'])
        tail_pairs = Batch.from_data_list(tail_list, follow_batch=['edge_index','line_graph_edge_index'])
        label = torch.cat(label_list, dim=0)
        return head_pairs, tail_pairs, label


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



# class DrugDataset4(Dataset):
#     def __init__(self, data_df, drug_graph,args):
#         self.data_df = data_df
#         self.drug_graph = drug_graph
#         self.args = args
#
#     def __len__(self):
#         return len(self.data_df)
#
#     def __getitem__(self, index):
#         return self.data_df.iloc[index]
#
#     # def _create_b_graph(self,edge_index,x_h, x_t):
#     #     return BipartiteData(edge_index,x_h,x_t)
#
#     def collate_fn(self, batch):
#         head_list = []
#         tail_list = []
#         label_list = []
#
#         for row in batch:
#             Drug1_ID, Drug2_ID, Y = row['ID1'], row['ID2'], row['Y']
#             h_graph = self.drug_graph.get(Drug1_ID)
#             t_graph = self.drug_graph.get(Drug2_ID)
#
#             if (h_graph.x.shape[0]==1 or h_graph.edge_index.shape[0]==0 or h_graph.edge_attr.shape[0]==0 or h_graph.line_graph_edge_index.shape[0]==0) or (t_graph.x.shape[0]==1 or t_graph.edge_index.shape[0]==0 or t_graph.edge_attr.shape[0]==0 or t_graph.line_graph_edge_index.shape[0]==0) :
#                 continue
#
#             head_list.append(h_graph)
#             tail_list.append(t_graph)
#             label_list.append(torch.FloatTensor([Y]))
#
#         head_pairs = Batch.from_data_list(head_list, follow_batch=['edge_index','line_graph_edge_index'])
#         tail_pairs = Batch.from_data_list(tail_list, follow_batch=['edge_index','line_graph_edge_index'])
#         label = torch.cat(label_list, dim=0)
#         return head_pairs, tail_pairs, label
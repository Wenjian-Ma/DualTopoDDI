import torch#mine+SA-DDI substructure+cl
import torch.nn as nn
# from torch_geometric.nn.inits import glorot
from torch_scatter import scatter#,scatter_softmax
import torch.nn.functional as F
from torch_geometric.utils import degree
from torch_geometric.utils import softmax
from torch_geometric.nn.conv import GraphConv
from torch_geometric.nn import global_add_pool,global_mean_pool,GATConv
# from torch_geometric.utils import to_dense_batch


class GlobalAttentionPool(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.conv = GraphConv(hidden_dim, 1)

    def forward(self, x, edge_index, batch):
        x_conv = self.conv(x, edge_index)
        scores = softmax(x_conv, batch, dim=0)
        gx = global_add_pool(x * scores, batch)

        return gx


class LinearBlock(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.snd_n_feats = 6 * n_feats
        self.lin1 = nn.Sequential(
            nn.BatchNorm1d(n_feats),
            nn.Linear(n_feats, self.snd_n_feats),
        )
        self.lin2 = nn.Sequential(

            nn.BatchNorm1d(self.snd_n_feats),
            nn.PReLU(),
            # nn.Dropout(0.2),
            nn.Linear(self.snd_n_feats, self.snd_n_feats),
        )
        self.lin3 = nn.Sequential(

            nn.BatchNorm1d(self.snd_n_feats),
            nn.PReLU(),
            # nn.Dropout(0.2),
            nn.Linear(self.snd_n_feats, self.snd_n_feats),
        )
        self.lin4 = nn.Sequential(

            nn.BatchNorm1d(self.snd_n_feats),
            nn.PReLU(),
            # nn.Dropout(0.2),
            nn.Linear(self.snd_n_feats, self.snd_n_feats)
        )
        self.lin5 = nn.Sequential(

            nn.BatchNorm1d(self.snd_n_feats),
            nn.PReLU(),
            # nn.Dropout(0.2),
            nn.Linear(self.snd_n_feats, n_feats)
        )

    def forward(self, x):
        x = self.lin1(x)
        x = (self.lin3(self.lin2(x)) + x) / 2
        x = (self.lin4(x) + x) / 2
        x = self.lin5(x)

        return x


class GlobalAttentionPool(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.conv = GraphConv(hidden_dim, 1)

    def forward(self, x, edge_index, batch):
        x_conv = self.conv(x, edge_index)
        scores = softmax(x_conv, batch, dim=0)
        gx = global_add_pool(x * scores, batch)

        return gx

class substructure_SA(nn.Module):
    def __init__(self, hidden_dim,args):
        super().__init__()
        # self.node_conv = GraphConv(hidden_dim, hidden_dim)
        #                                # nn.Tanh(),
        #                                # nn.Linear(hidden_dim,1)
        #
        # self.line_conv = GraphConv(hidden_dim,hidden_dim)

        self.node_pool = GlobalAttentionPool(hidden_dim)
        self.line_pool = GlobalAttentionPool(hidden_dim)

        self.tanh = nn.Tanh()

        self.node_Linear = nn.Linear(hidden_dim,1)
        self.line_Linear = nn.Linear(hidden_dim, 1)
                                       # nn.Tanh(),
                                       # nn.Linear(hidden_dim,1)

        self.node_lin_block = LinearBlock(hidden_dim)#
        self.line_lin_block = LinearBlock(hidden_dim)#
        self.args = args
    def forward(self, node_list,line_list,data):

        node_alpha_list = []
        line_alpha_list = []
        g_node_list = []
        g_line_list = []
        for i in range(self.args.n_iter):
            g_node = self.tanh(self.node_pool(node_list[i], data.edge_index,data.batch))
            g_line = self.tanh(self.line_pool(line_list[i], data.line_graph_edge_index,data.edge_index_batch))

            g_node_list.append(g_node)
            g_line_list.append(g_line)

            node_alpha = self.node_Linear(g_node)
            line_alpha = self.line_Linear(g_line)
            node_alpha_list.append(node_alpha)
            line_alpha_list.append(line_alpha)
        node_alpha_list = torch.stack(node_alpha_list,dim=-1)#.squeeze()
        line_alpha_list = torch.stack(line_alpha_list, dim=-1)#.squeeze()
        node_attn = torch.softmax(node_alpha_list,dim=-1)#.unsqueeze(1)
        line_attn = torch.softmax(line_alpha_list,dim=-1)#.unsqueeze(1)

        node_attn = node_attn.repeat_interleave(degree(data.batch, dtype=data.batch.dtype), dim=0)
        line_attn = line_attn.repeat_interleave(degree(data.edge_index_batch, dtype=data.edge_index_batch.dtype), dim=0)

        # a = torch.stack(node_list, dim=-1)
        # b = node_attn * torch.stack(node_list, dim=-1)
        out_node = self.node_lin_block((node_attn * torch.stack(node_list,dim=-1)).sum(-1) + node_list[-1])#
        out_line = self.line_lin_block((line_attn * torch.stack(line_list,dim=-1)).sum(-1) + line_list[-1])

        return out_node,out_line,g_node_list,g_line_list

class MH_substructure_SA(nn.Module):
    def __init__(self, hidden_dim,args):
        super().__init__()
        self.args = args
        self.multi_heads_sub_sa = nn.ModuleList()
        for i in range(self.args.heads):
            self.multi_heads_sub_sa.append(substructure_SA(hidden_dim, args=args))
        self.lin_node = nn.Linear(self.args.heads * hidden_dim,hidden_dim)
        self.lin_line = nn.Linear(self.args.heads * hidden_dim, hidden_dim)
    def forward(self,node_list,line_list,data):

        x_node_list = []
        x_line_list = []
        mh_g_node_list = []
        mh_g_line_list = []
        for sa_layer in self.multi_heads_sub_sa:
            x_node,x_line,g_node_list,g_line_list = sa_layer(node_list,line_list,data)
            x_node_list.append(x_node)
            x_line_list.append(x_line)
            mh_g_node_list.extend(g_node_list)
            mh_g_line_list.extend(g_line_list)

        x_node_list = torch.concat(x_node_list,1)
        x_line_list = torch.concat(x_line_list,1)

        x_node_output = self.lin_node(x_node_list)
        x_line_output = self.lin_line(x_line_list)
        return x_node_output,x_line_output,mh_g_node_list,mh_g_line_list


# class InterGraphAttention_line(nn.Module):
#     def __init__(self, input_dim):
#         super().__init__()
#         self.input_dim = input_dim
#         self.inter = GATConv((input_dim, input_dim), 32, 2)
#         self.prelu = nn.PReLU()
#     def forward(self, h_data, t_data, b_graph):
#         edge_index = b_graph.edge_index
#         h_input = self.prelu((h_data.edge_attr))
#         t_input = self.prelu((t_data.edge_attr))
#         t_rep = self.inter((h_input, t_input), edge_index)
#         h_rep = self.inter((t_input, h_input), edge_index[[1, 0]])
#
#         return h_rep, t_rep


class DMPNN(nn.Module):
    def __init__(self, n_feats,args=None):
        super().__init__()
        self.args = args

        self.node_attn_w_i = nn.Linear(n_feats, n_feats)
        self.node_attn_w_j = nn.Linear(n_feats, n_feats)
        self.node_attn_w_ij = nn.Linear(n_feats, n_feats)


        self.line_attn_w_ij = nn.Linear(n_feats, n_feats)
        self.line_attn_w_ik = nn.Linear(n_feats, n_feats)
        self.line_attn_w_i = nn.Linear(n_feats, n_feats)

        self.d_k = torch.sqrt(torch.Tensor([n_feats])).to('cuda:'+self.args.device)

        self.updata_node = nn.Sequential(
            nn.Linear(n_feats, n_feats),

            nn.BatchNorm1d(n_feats),
            nn.PReLU()
        )
        self.updata_line = nn.Sequential(
            nn.Linear(n_feats, n_feats),

            nn.BatchNorm1d(n_feats),
            nn.PReLU()
        )


    def forward(self, data):
        edge_index = data.edge_index
        line_graph_edge_index = data.line_graph_edge_index

        node_fea_attn_i = self.node_attn_w_i(data.x)
        node_fea_attn_j = self.node_attn_w_j(data.x)
        node_fea_attn_ij = self.node_attn_w_ij(data.edge_attr)

        a_ij = (node_fea_attn_i[edge_index[1, :]] * (node_fea_attn_j[edge_index[0, :]] + node_fea_attn_ij)).sum(
            -1) / self.d_k

        alpha_ij = softmax(a_ij.unsqueeze(1), data.edge_index[1, :], dim=0)

        message_node = scatter(alpha_ij * (data.x[edge_index[0, :]] + data.edge_attr), edge_index[1, :],
                               dim_size=data.x.size(0), dim=0, reduce='add')
        data.x = self.updata_node(message_node + data.x)

        #########################################

        line_fea_attn_ij = self.line_attn_w_ij(data.edge_attr)
        line_fea_attn_ik = self.line_attn_w_ik(data.edge_attr)
        line_fea_attn_i = self.line_attn_w_i(data.x)

        a_i = (line_fea_attn_ij[line_graph_edge_index[1, :]] * (
                line_fea_attn_ik[line_graph_edge_index[0, :]] + line_fea_attn_i[
            edge_index[0, :][line_graph_edge_index[1, :]]])).sum(-1) / self.d_k

        alpha_i = softmax(a_i.unsqueeze(1), data.line_graph_edge_index[1, :], dim=0)

        message_line = scatter(alpha_i * (data.edge_attr[line_graph_edge_index[0, :]] + data.x[
            edge_index[0, :][line_graph_edge_index[1, :]]]), line_graph_edge_index[1, :],
                               dim_size=data.edge_attr.size(0), dim=0, reduce='add')

        data.edge_attr = self.updata_line(message_line + data.edge_attr)

        return data


class DrugEncoder(torch.nn.Module):
    def __init__(self, in_dim, edge_in_dim, hidden_dim=64, n_iter=10,args=None):
        super().__init__()

        if args.dataset == 'drugbank' or args.dataset == 'ChChMiner' or args.dataset=='MMDDI' or args.dataset=='AUC_FC':
            self.mlp_node = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.PReLU(),
                #nn.Dropout(0.2)#这行去掉 for drugbank
            )

            self.mlp_edge = nn.Sequential(
                nn.Linear(edge_in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.PReLU(),
                #nn.Dropout(0.2)#这行去掉 for drugbank
            )
        elif args.dataset == 'twosides' or args.dataset == 'ZhangDDI' or args.dataset == 'DeepDDI':
            self.mlp_node = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.PReLU(),
                nn.Dropout(0.2)#这行去掉 for drugbank
            )

            self.mlp_edge = nn.Sequential(
                nn.Linear(edge_in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.PReLU(),
                nn.Dropout(0.2)#这行去掉 for drugbank
            )



        self.Cross_MPNN = nn.ModuleList()
        for _ in range(n_iter):
            self.Cross_MPNN.append(DMPNN(hidden_dim,args=args))
            # self.Cross_MPNN.append(GraphConv(hidden_dim,hidden_dim ))

        # self.substructure_SA = substructure_SA(hidden_dim,args=args)
        self.mh_sub_sa = MH_substructure_SA(hidden_dim,args=args)


    def forward(self, data):
        data.x = self.mlp_node(data.x)
        data.edge_attr = self.mlp_edge(data.edge_attr)



        node_list = []
        line_list = []



        for nn in self.Cross_MPNN:

            data = nn(data)


            # data.x = nn(data.x,data.edge_index)

            node_list.append(data.x)
            line_list.append(data.edge_attr)

        x_node,x_line,mh_g_node_list,mh_g_line_list = self.mh_sub_sa(node_list,line_list,data)

        return x_node,x_line,mh_g_node_list,mh_g_line_list

        # return data.x, data.edge_attr, mh_g_node_list, mh_g_line_list

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




        return s_h,s_t,s_h_line,s_t_line

class nnModel(torch.nn.Module):
    def __init__(self, in_dim, edge_in_dim, hidden_dim=64, n_iter=10,args=None):
        super(nnModel, self).__init__()

        self.drug_encoder = DrugEncoder(in_dim, edge_in_dim, hidden_dim, n_iter=n_iter,args=args)

        self.hma = hierarchical_mutual_attn(hidden_dim)




        self.rmodule = nn.Embedding(963, hidden_dim)#86 963

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



class nnModel2(torch.nn.Module):
    def __init__(self, in_dim, edge_in_dim, hidden_dim=64, n_iter=10,args=None):
        super(nnModel2, self).__init__()

        self.drug_encoder = DrugEncoder(in_dim, edge_in_dim, hidden_dim, n_iter=n_iter,args=args)

        self.hma = hierarchical_mutual_attn(hidden_dim)

        self.sigmoid = nn.Sigmoid()



        #####################

        if args.dataset == 'drugbank' or args.dataset == 'ChChMiner':
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
                nn.Linear(hidden_dim * 4, hidden_dim),
            )
    def forward(self, triples):
        h_data, t_data = triples



        x_node_h,x_line_h,h_g_node_list,h_g_line_list = self.drug_encoder(h_data)#子结构提取模块 + 子注意力模块的输出应该是原子尺度
        x_node_t, x_line_t,t_g_node_list,t_g_line_list = self.drug_encoder(t_data)


        ##################

        x_node_h,x_node_t,x_line_h,x_line_t = self.hma(x_node_h,x_node_t,x_line_h,x_line_t,h_data, t_data)



        #################################################

        rep = torch.cat([x_node_h,x_line_h, x_node_t,x_line_t], dim=-1)



        logit = (self.lin(rep)).sum(-1)  # 97.5
        #####################################################




        output = self.sigmoid(logit)

        return output,h_g_node_list,h_g_line_list,t_g_node_list,t_g_line_list


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

        x_node_h,x_node_t,x_line_h,x_line_t = self.hma(x_node_h,x_node_t,x_line_h,x_line_t,h_data, t_data)



        #################################################

        rep = torch.cat([x_node_h,x_line_h, x_node_t,x_line_t], dim=-1)



        logit = self.lin(rep)
        #####################################################




        output = logit

        return output,h_g_node_list,h_g_line_list,t_g_node_list,t_g_line_list


class nnModel_AUC_FC(torch.nn.Module):
    def __init__(self, in_dim, edge_in_dim, hidden_dim=64, n_iter=10,args=None):
        super(nnModel_AUC_FC, self).__init__()

        self.drug_encoder = DrugEncoder(in_dim, edge_in_dim, hidden_dim, n_iter=n_iter,args=args)

        self.hma = hierarchical_mutual_attn(hidden_dim)

        #####################

        if args.dataset == 'drugbank' or args.dataset == 'ChChMiner' or args.dataset == 'MMDDI' or args.dataset == 'AUC_FC':
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
                nn.Linear(hidden_dim * 4, hidden_dim),
            )
    def forward(self, triples):
        h_data, t_data = triples



        x_node_h,x_line_h,h_g_node_list,h_g_line_list = self.drug_encoder(h_data)#子结构提取模块 + 子注意力模块的输出应该是原子尺度
        x_node_t, x_line_t,t_g_node_list,t_g_line_list = self.drug_encoder(t_data)


        ##################

        x_node_h,x_node_t,x_line_h,x_line_t = self.hma(x_node_h,x_node_t,x_line_h,x_line_t,h_data, t_data)



        #################################################

        rep = torch.cat([x_node_h,x_line_h, x_node_t,x_line_t], dim=-1)



        logit = self.lin(rep)
        #####################################################




        output = logit.sum(-1)

        return output,h_g_node_list,h_g_line_list,t_g_node_list,t_g_line_list
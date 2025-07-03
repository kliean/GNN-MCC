import time

import math
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
from sklearn.utils.extmath import weighted_mode
from torch import nn
from torch.nn import init
from torch.distributions import Normal, Independent
from torch.nn.functional import softplus
from torch_geometric.nn import SAGEConv, GATConv, GCNConv
from torch_geometric.utils import k_hop_subgraph
from args import args
from torchvision.models import resnet18
from transformers import BertModel, BertConfig


class MGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, gnn_type='sage', gnn_num_layer=2, gnn_out_dim=512, activation='relu', aggr='max'):
        super().__init__()
        self.aggr = aggr
        self.gnn_type = gnn_type
        self.gnn_out_dim = gnn_out_dim
        self.p = args.dropout
        self.heads = args.heads
        self.center = nn.Parameter(torch.randn(num_classes, self.gnn_out_dim))
        self.conv1 = SAGEConv(in_dim, hidden_dim, aggr=self.aggr)
        self.conv2 = SAGEConv(hidden_dim, self.gnn_out_dim, aggr=self.aggr)
        self.conv3 = SAGEConv(self.gnn_out_dim, num_classes, aggr=self.aggr)

        self.num_layers = gnn_num_layer
        self.batch_norm = nn.BatchNorm1d(self.gnn_out_dim)
        activation_name = activation
        activation_by_name = {'relu': nn.ReLU(), 'prelu': nn.PReLU()}
        activation = activation_by_name[activation_name]
        gnn_layers = []
        if self.gnn_type == 'sage':
            input_dim, output_dim = in_dim, hidden_dim
            for _ in range(self.num_layers - 1):
                gnn_layers += [
                    SAGEConv(input_dim, output_dim, aggr=self.aggr),
                    nn.BatchNorm1d(output_dim),
                    # DyT(output_dim),
                    # activation,
                    nn.Dropout(self.p),
                ]
                input_dim = output_dim
            gnn_layers += [SAGEConv(hidden_dim, self.gnn_out_dim, aggr=self.aggr)]

        if self.gnn_type == 'gcn':
            input_dim, output_dim = in_dim, hidden_dim
            for _ in range(self.num_layers - 1):
                gnn_layers += [
                    GCNConv(input_dim, output_dim),
                    nn.BatchNorm1d(output_dim),
                    # DyT(output_dim),
                    activation,
                    nn.Dropout(self.p),
                ]
                input_dim = output_dim
            gnn_layers += [GCNConv(hidden_dim, self.gnn_out_dim)]

        if self.gnn_type == 'gat':
            input_dim, output_dim = in_dim, hidden_dim
            for _ in range(self.num_layers - 1):
                gnn_layers += [
                    GATConv(input_dim, output_dim, heads=self.heads),
                    nn.BatchNorm1d(output_dim * self.heads),
                    activation,
                    nn.Dropout(self.p),
                ]
                input_dim = output_dim * self.heads
            gnn_layers += [GATConv(input_dim, self.gnn_out_dim, heads=1)]
        self.gnn_layers = nn.ModuleList(gnn_layers)

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        for layer in self.gnn_layers:
            if isinstance(layer, torch_geometric.nn.SAGEConv):
                if hasattr(layer, 'lin'):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                if hasattr(layer, 'att'):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, data, edge_weight=None):
        x, edge_index = data.x, data.edge_index
        for layer in self.gnn_layers:
            if isinstance(layer, torch_geometric.nn.MessagePassing):
                x = layer(x, edge_index, edge_weight)
            else:
                x = layer(x)

        return  x

class Classifier(nn.Module):
    def __init__(self, in_dim, num_classes, type='mlp'):
        super().__init__()
        h_dim = 128
        if type == 'mlp':
            self.cls = nn.Linear(in_dim, num_classes)
            self.cls = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, num_classes)
        )
    def forward(self, x):
        return self.cls(x)

class ContrastiveAttentionCompensation(nn.Module):
    def __init__(self, feat_dim, num_classes, tau=1.0, temperature=0.1, batch_size=200):
        super().__init__()
        self.W_q = nn.Linear(feat_dim, feat_dim)
        self.W_k = nn.Linear(feat_dim, feat_dim)

    def forward(self, h1, h2, pseudo_labels=None):
        h1 = self.W_q(h1)
        h2 = self.W_k(h2)

        q = h1.unsqueeze(1)  # (B,1,D)
        k = h2.unsqueeze(0)  # (1,B,D)
        start_time = time.time()
        attn_logits = torch.sum(q * k, dim=-1) / math.sqrt(h1.shape[-1])  # (B,B)
        end_time = time.time()
        attn = attn_logits

        soft_text_attn = F.softmax(attn, dim=-1)
        soft_img_attn = F.softmax(attn, dim=0)

        fused_feat_h1 = torch.matmul(soft_text_attn, h2) + h1
        fused_feat_h2 = torch.matmul(soft_img_attn.T, h1) + h2

        return fused_feat_h1, fused_feat_h2, attn, end_time - start_time


class MMAG(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, gnn_type='sage', gnn_num_layer=2, gnn_out_dim=512, sim_tau=0.1,
                 activation='prelu', aggr='max'):
        super(MMAG, self).__init__()
        print(f'gnn type: {gnn_type}')
        self.sim_tau = sim_tau

        self.MGNN = MGNN(in_dim, hidden_dim, num_classes, gnn_type, gnn_num_layer, gnn_out_dim)

        z_dim, node_dim = gnn_out_dim * 4, gnn_out_dim * 2

        self.pairs_mlp = nn.Linear(z_dim, z_dim)
        self.pairs_cls = nn.Linear(z_dim, num_classes)
        self.edge_agreed_mlp = nn.Linear(z_dim, 2)

        h_dim = gnn_out_dim * 1
        self.bn = nn.BatchNorm1d(gnn_out_dim * 1)
        self.node_bn = nn.LayerNorm(h_dim)
        self.pair_bn = nn.LayerNorm(z_dim)

        self.node_cls = nn.Linear(h_dim, num_classes)

        self.node_mlp = nn.Linear(node_dim, h_dim)


    def forward(self, data, pairs_idx=None, h1_compensate=None, h2_compensate=None, test=False, edge_weight=None):
        src_nodes = pairs_idx[0]
        dst_nodes = pairs_idx[1]


        # GNN layers
        gnn_x = self.MGNN(data)
        # gnn_x = gnn_x[rel_node]

        bn_gnn_x = self.bn(gnn_x)

        x = torch.cat([bn_gnn_x, bn_gnn_x], -1)
        x[src_nodes] = torch.cat([h1_compensate, bn_gnn_x[src_nodes]], -1)
        x[dst_nodes] = torch.cat([h2_compensate, bn_gnn_x[dst_nodes]], -1)

        z = torch.cat([x[src_nodes], x[dst_nodes]], -1)

        z = self.pairs_mlp(z)
        # z = F.dropout(z, p=0.2)
        z = self.pair_bn(z) # layernorm
        z = F.relu(z)
        pairs_logit = self.pairs_cls(z)

        pairs_consistency_logit = self.edge_agreed_mlp(z)

        x1 = self.node_mlp(x)
        x1 = self.node_bn(x1)
        node_logit = self.node_cls(F.relu(x1))
        # node_logit = self.node_cls(x)
        return node_logit, pairs_logit, pairs_consistency_logit, x, gnn_x


class UniMModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, gnn_type='sage', gnn_num_layer=2, gnn_out_dim=512, activation='prelu', aggr='max'):
        super(UniMModel, self).__init__()
        self.MGNN = MGNN(in_dim, hidden_dim, num_classes, gnn_type, gnn_num_layer, gnn_out_dim)

        self.node_mlp = nn.Sequential(
            nn.Linear(gnn_out_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, data, edge_weight=None):
        # GNN layers
        x = self.MGNN(data)

        return self.node_mlp(x), x

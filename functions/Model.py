import os
from PIL import Image
import time
import copy
import json
from math import ceil, sqrt
import random
import pandas as pd
import numpy as np
from random import randint

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import Sequential, Linear, ReLU, LeakyReLU
import torch.nn.functional as F
# from torchvision.models.resnet import model_urls


from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch_geometric.data as td
import torch_geometric.transforms as T
from torch_geometric.data import DataListLoader
from torch_geometric.nn import SAGEConv, GINConv, DenseGCNConv, GCNConv, GATConv, EdgeConv
# from torch_geometric.utils import dense_to_sparse, dropout_adj,true_positive, true_negative, false_positive, false_negative, precision, recall, f1_score
from torch_geometric.nn import knn_graph, dense_diff_pool, global_add_pool, global_mean_pool, DataParallel
from torch_geometric.nn import JumpingKnowledge as jp
from torch_geometric.utils import to_dense_batch
from torch_geometric.data import Data

from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn import metrics

# def graph_new(x):
#     pre_activation = x.transpose(0, 1)
#     # Create a zero vector with the same number of columns
#     zero_vector = torch.zeros(1, pre_activation.size(1))
#     # Concatenate the zero vector to the initial tensor
#     # 确保 pre_activation 和 zero_vector 都在相同的设备上
#     device = pre_activation.device  # 获取 pre_activation 所在设备
#     zero_vector = zero_vector.to(device)  # 将 zero_vector 移动到同一设备
#
#     # 然后执行拼接
#     new_node = torch.cat((pre_activation, zero_vector), dim=0)
#
#     # Example graph structure
#     # edge_index = torch.tensor([[0, 1, 2, 3, 4, 5],
#     #                            [2, 5, 3, 5, 1, 0]], dtype=torch.long)
#     edge_index = torch.randint(0, 5, (2, 24))
#
#     # edge_attr = torch.tensor([15, 1.5, 0.1, 5.0, 0.1, 5.0], dtype=torch.float)
#     random_tensor = torch.randn(20, 1,dtype=torch.float)
#     edge_attr = torch.clamp(random_tensor, min=0, max=2)
#
#     # Create a PyG data object
#     data = Data(x=new_node, edge_index=edge_index, edge_attr=edge_attr)
#
#     data.added_node_index = new_node.size(0) - 1  # 新增节点的索引
#
#     return data


# 定义Graph类
class Graph:
    def __init__(self, node_features, num_edges):
        self.x = node_features  # 节点特征
        self.edge_index = None  # 边连接关系
        self.added_node_index = node_features.size(0) - 1  # 新增节点的索引

    def update_node_features(self, new_features):
        self.x[:-1] = new_features.T  # 更新节点特征


# 定义 graph_new 函数
def graph_new(node_features):
    # 转置节点特征，使其符合 PyG 的要求
    pre_activation = node_features.transpose(0, 1)

    # 创建一个零向量用于新增节点
    zero_vector = torch.zeros(1, pre_activation.size(1), device=node_features.device)

    # 创建一个新节点特征，包括旧节点和新节点
    new_node = torch.cat((pre_activation, zero_vector), dim=0)

    # 老节点数目和新节点索引
    num_old_nodes = new_node.size(0) - 1
    new_node_index = num_old_nodes

    # 仅连接每个老节点到新节点
    edge_index = torch.vstack((
        torch.arange(num_old_nodes),  # 老节点的索引（0, 1, 2, ...）
        torch.full((num_old_nodes,), new_node_index)  # 目标节点为新节点
    ))

    return new_node, edge_index  # 返回节点特征和边索引


# def graph_new(node_features):
#     # 转置节点特征，使其符合 PyG 的要求
#     pre_activation = node_features.transpose(0, 1)
#
#     # 创建一个零向量用于新增节点
#     zero_vector = torch.zeros(1, pre_activation.size(1), device=node_features.device)
#
#     # 创建一个新节点特征，包括旧节点和新节点
#     new_node = torch.cat((pre_activation, zero_vector), dim=0)
#
#     # 老节点数目和新节点索引
#     num_old_nodes = new_node.size(0) - 1
#     new_node_index = num_old_nodes
#
#     # 仅连接每个老节点到新节点
#     edge_index = torch.vstack((
#         torch.arange(num_old_nodes),  # 老节点的索引（0, 1, 2, ...）
#         torch.full((num_old_nodes,), new_node_index)  # 目标节点为新节点
#     ))
#
#     # 边属性初始化为可学习的参数
#     edge_attr = nn.Parameter(torch.randn(edge_index.size(1), 1))  # 或者使用其他初始化方式
#     edge_attr.data = torch.clamp(edge_attr.data, min=0, max=2)  # 限制边属性的范围（可选）
#
#     # 创建 PyG 数据对象
#     data = Data(x=new_node, edge_index=edge_index, edge_attr=edge_attr)
#
#     # 存储新增节点的索引
#     data.added_node_index = new_node.size(0) - 1  # 新增节点的索引
#
#     return data


class AdaptiveEdgeWeightGNN(nn.Module):
    def __init__(self, in_channels, out_channels, num_edges, fixed_values):
        super(AdaptiveEdgeWeightGNN, self).__init__()
        self.gcn1 = GCNConv(in_channels, out_channels)
        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.edge_attr = nn.Parameter(torch.tensor(fixed_values, dtype=torch.float32).view(-1, 1))

    def forward(self, x, edge_index, edge_attr = None):
        device = x.device  # 假设 edge_index 已经在正确的设备上
        edge_index = edge_index.to(device)
        if edge_attr is None:
            edge_attr = self.edge_attr.to(device) # 使用模型的边属性
        else:
            edge_attr = edge_attr.to(device)

        # 通过GCN传播节点特征，传入可学习的边权重
        x = self.gcn1(x, edge_index, edge_attr)
        return x

class MultiHeadFullyConnected(torch.nn.Module):
    def __init__(self, in_features, out_features, num_heads):
        super(MultiHeadFullyConnected, self).__init__()
        self.heads = torch.nn.ModuleList()
        for _ in range(num_heads):
            self.heads.append(torch.nn.Linear(in_features, out_features))

    def forward(self, x):
        outputs = [head(x) for head in self.heads]
        return torch.stack(outputs, dim=-1)

    def reset_parameters(self):
        for head in self.heads:
            nn.init.xavier_uniform_(head.weight)
            if head.bias is not None:
                nn.init.zeros_(head.bias)

class MultiNet(nn.Module):
    def __init__(self, num_classes):
        super(MultiNet, self).__init__()

        self.nhid = 256
        self.feature = 64
        self.num_classes = num_classes

        nn1 = Sequential(
            Linear(self.feature, self.nhid),
            LeakyReLU(),
            Linear(self.nhid, self.nhid)
        )  # , ReLU(), Linear(self.nhid, self.nhid))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(self.nhid)

        nn2 = Sequential(
            Linear(self.nhid, self.nhid),
            LeakyReLU(),
            Linear(self.nhid, self.nhid)
        )  # , ReLU(), Linear(self.nhid, self.nhid))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(self.nhid)


        nn3 = Sequential(
            Linear(self.nhid, self.nhid),
            LeakyReLU(),
            Linear(self.nhid, self.nhid)
        )  # , ReLU(), Linear(self.nhid, self.nhid))
        self.conv3 = GINConv(nn3)

        self.jpn = jp('max')  # 'max'
        # self.jpn = jp('lstm',self.nhid,3)

        self.fc1 = Linear(self.nhid, self.nhid // 2)
        self.fc2 = Linear(self.nhid // 2, self.nhid // 4)
        self.fc3 = Linear(self.nhid // 4, num_classes )

        # self.ReLU  = nn.ReLU ()
        # self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        h = []
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # if self.training:
        #     noise = torch.randn_like(x) * 0.5  # 可根据需要调整噪声强度
        #     x = x + noise

        x = F.leaky_relu(self.conv1(x, edge_index))
        h1 = x
        x = self.bn1(x)
        h.append(x)

        x = F.leaky_relu(self.conv2(x, edge_index))
        h2 = x
        x = self.bn2(x)
        h.append(x)

        x = F.leaky_relu(self.conv3(x, edge_index))
        h3 = x

        h.append(x)
        # print('x3')
        # x = self.jpn([h1,h2,h3])
        x = self.jpn(h)  # , h4])

        # attn_x = copy.deepcopy(x)
        # attn_x = global_add_pool(attn_x, batch, 10)
        select_index = global_sort_pool(x, batch, 20)
        x = global_add_pool(x, batch)

        x = self.fc1(x)
        # x = F.relu(F.dropout(x, p=0.5, training=self.training))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)

        # x_ReLU = self.ReLU (x)
        # x_out = self.sigmoid(x_ReLU)
        x_out = x

        return x_out , select_index

class Net(nn.Module):
    def __init__(self, input_dim, feature_num, fixed_values):
        super(Net, self).__init__()
        self.simple_nn = MultiNet(input_dim)
        self.gcn_layer = AdaptiveEdgeWeightGNN(feature_num, feature_num, input_dim, fixed_values)

        # 初始化图结构
        self.graph = None

    def forward(self, x):
        # 前向传播：传统神经网络
        pre_activation,select_index = self.simple_nn(x)

        # 只在第一次调用时创建图结构
        if self.graph is None:
            node_features, edge_index = graph_new(pre_activation)
            self.graph = Data(x=node_features, edge_index=edge_index)
            self.graph.added_node_index = node_features.size(0) - 1  # 新增节点的索引

        # 更新图中的节点特征
        self.graph.x[:-1] = pre_activation.T  # 更新节点特征

        # 构图并进行图传播
        gcn_out = self.gcn_layer(self.graph.x, self.graph.edge_index, self.graph.edge_attr)
        added_node_features = gcn_out[self.graph.added_node_index].unsqueeze(0)
        self.graph.edge_attr = self.gcn_layer.edge_attr
        edge_index = self.graph.edge_index
        index1 = edge_index[0]
        index2 = edge_index[1]

        # 最终分类
        return pre_activation , select_index, added_node_features, self.graph


def global_sort_pool(x, batch, k):
    r"""The global pooling operator from the `"An End-to-End Deep Learning
    Architecture for Graph Classification"
    <https://www.cse.wustl.edu/~muhan/papers/AAAI_2018_DGCNN.pdf>`_ paper,
    where node features are sorted in descending order based on their last
    feature channel. The first :math:`k` nodes form the output of the layer.
    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        batch (LongTensor): Batch vector :math:`\mathbf{b} \in {\{ 0, \ldots,
            B-1\}}^N`, which assigns each node to a specific example.
        k (int): The number of nodes to hold for each graph.
    :rtype: :class:`Tensor`
    """
    fill_value = x.min().item() - 1
    batch_x, _ = to_dense_batch(x, batch, fill_value)
    #     print(batch_x)
    B, N, D = batch_x.size()

    copy_all = []  # copy.deepcopy(batch_x).tolist()
    for i in batch_x:
        copy_all.append(i.tolist())
    _, perm = batch_x[:, :, -1].sort(dim=-1, descending=True)
    arange = torch.arange(B, dtype=torch.long, device=perm.device) * N
    perm = perm + arange.view(-1, 1)

    batch_x = batch_x.view(B * N, D)
    batch_x = batch_x[perm]
    batch_x = batch_x.view(B, N, D)

    if N >= k:
        batch_x = batch_x[:, :k].contiguous()
        copy_select = batch_x.tolist()
        select_index = []
        for ori_graph, k_graph in zip(copy_all, copy_select):
            node_index = []
            for node in k_graph:
                node_index.append(ori_graph.index(node))
            select_index.append(node_index)
    else:
        expand_batch_x = batch_x.new_full((B, k - N, D), fill_value)
        batch_x = torch.cat([batch_x, expand_batch_x], dim=1)

    return torch.Tensor(select_index).cuda()

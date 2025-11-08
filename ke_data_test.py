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
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer
import torch.nn as nn
import torch.nn.init as init
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
from torch.nn import Sequential, Linear, ReLU
import torch.nn.functional as F
from torchvision.models.resnet import model_urls

from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch_geometric.data as td
import torch_geometric.transforms as T

from torch_geometric.data import DataListLoader, Data
from torch_geometric.nn import SAGEConv, GINConv, DenseGCNConv, GCNConv, GATConv, EdgeConv
# from torch_geometric.utils import dense_to_sparse, dropout_adj,true_positive, true_negative, false_positive, false_negative, precision, recall, f1_score
from torch_geometric.nn import knn_graph, dense_diff_pool, global_add_pool, global_mean_pool, DataParallel
from torch_geometric.nn import JumpingKnowledge as jp
from torch_geometric.utils import to_dense_batch

from itertools import chain

from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, accuracy_score, hamming_loss, roc_auc_score, f1_score, recall_score



##############################################################################################
# Customized
# from functions.model_architecture import Net, global_sort_pool
from multi_just import Net, global_sort_pool
from functions.utils import generate_dataset, oversampling, majority_vote
##############################################################################################
import csv

def read_csv_to_dict(file_path):
    with open(file_path, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        data_dict = {}
        for row in reader:
            for key, value in row.items():
                if key in data_dict:
                    data_dict[key].append(value)
                else:
                    data_dict[key] = [value]
    return data_dict

def read_csv_gene(file_path):
    with open(file_path, mode='r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)
        data_dict = {header: [] for header in headers}
        for row in reader:
            for header, value in zip(headers, row):
                data_dict[header].append(value)
    return data_dict

def read_csv_pathway(file_path):
    with open(file_path, mode='r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        data_dict = {}
        current_identifier = None
        keys = []
        for row in reader:
            # Check if the row is an identifier row for "CAN", "Gene", or "Protein"
            if row and (row[0].startswith('CAN') or row[0].startswith('Gene') or row[0].startswith('Protein')):
                current_identifier = row[0]  # Set the current identifier
                keys = []  # Reset keys for the new identifier
            elif current_identifier:
                if not keys:  # If keys are not set, the current row is treated as keys
                    keys = [f"{key.strip()}_{current_identifier}" for key in row if key.strip()]
                else:  # Process values for the keys
                    for key, value in zip(keys, row):
                        if key in data_dict:
                            data_dict[key].append(value.strip())
                        else:
                            data_dict[key] = [value.strip()]
    return data_dict

def extract_before_delimiters(input_list):
    result = []
    for item in input_list:
        pos_colon = item.find(':')
        pos_tilde = item.find('~')

        if pos_colon != -1 and (pos_tilde == -1 or pos_colon < pos_tilde):
            result.append(item[:pos_colon])
        elif pos_tilde != -1:
            result.append(item[:pos_tilde])
        else:
            result.append(item)

    return result

# Replace 'path/to/your/first_csv_file.csv' and 'path/to/your/second_csv_file.csv' with your actual file paths
csv_file_gene = '/mnt/data16t/yuy/pathway/label/gene_list.csv'
csv_file_pathwawy ='/mnt/data16t/yuy/pathway/label/pathway_new.csv'

dict_gene = read_csv_gene(csv_file_gene)
dict_pathway = read_csv_pathway(csv_file_pathwawy)

###################################################################      Must define task and molecular profile name here (in order)   ######################################################################
pathway_task_list = ['Gene']
pathway = ['hsa04151']
##############################################################################################
for task in pathway_task_list:
    if task == 'CNA':
        Term_CAN = dict_pathway['Term_CAN']
        Term_CAN = extract_before_delimiters(Term_CAN)
        Genes_CAN = dict_pathway['Genes_CAN']
        CAN_dict = dict(zip(Term_CAN, Genes_CAN))
        for p in pathway:
            if p in CAN_dict:
                genes = CAN_dict[p]
    elif task == 'Gene':
        Term_Gene = dict_pathway['Term_Gene']
        Term_Gene = extract_before_delimiters(Term_Gene)
        Genes_Gene = dict_pathway['Genes_Gene']
        Gene_dict = dict(zip(Term_Gene, Genes_Gene))
        for p in pathway:
            if p in Gene_dict:
                genes = Gene_dict[p]
    elif task == 'Protein':
        Term_Protein = dict_pathway['Term_Protein']
        Term_Protein = extract_before_delimiters(Term_Protein)
        Genes_Protein = dict_pathway['Genes_Protein']
        Protein_dict = dict(zip(Term_Protein, Genes_Protein))
        for p in pathway:
            if p in Protein_dict:
                genes = Protein_dict[p]

gene_list = genes.split(',')
task_list = [task] * len(gene_list)
gene_list = [gene.strip() for gene in gene_list]
# task_list = ['Gene','Gene','Gene']
# gene_list = ['TP53','MUC16','EGFR']
top_gene_list = dict_gene['gene']
exclude_values = ['TNN']
hub_gene = ['ARAF', 'ASPM', 'BIRC6', 'CPEB3', 'CSMD3', 'CUL3', 'EED', 'EGFR', 'EPHA3', 'GPC5', 'GRIN2A', 'HGF', 'HIF1A', 'KRAS', 'LEPROTL1', 'MALAT1', 'MB21D2', 'MYCL', 'N4BP2', 'NOTCH1', 'NTRK2', 'PTPN13', 'PTPRD', 'PTPRT', 'RAD21', 'RB1', 'RBM10', 'RFWD3', 'SIRPA', 'SOS1', 'STRN', 'SUB1', 'TP53', 'USP44', 'ZNF479']
# gene_list = [gene for gene in set(top_gene_list) & set(gene_list) if gene not in exclude_values]
# 将top_gene_list中的基因位置映射到gene_list中
top_gene_dict = {gene: i for i, gene in enumerate(top_gene_list)}
filtered_gene_list = [gene for gene in gene_list if gene not in exclude_values]
# 根据top_gene_list的排序来筛选出gene_list中的前五个基因
sorted_genes_in_top = [gene for gene in filtered_gene_list if gene in top_gene_dict]
sorted_genes_in_top.sort(key=lambda x: top_gene_dict[x])
# 获取前五个基因
gene_list = sorted_genes_in_top[:5]
# 将 hub_gene 中的基因加入 top_5_genes 中，但前提是这些基因不在 top_5_genes 中且在 filtered_gene_list 中
for gene in hub_gene:
    if gene in filtered_gene_list and gene not in gene_list:
        gene_list.append(gene)
index_hub = []
for gene in gene_list:
    if gene in hub_gene:
        index = gene_list.index(gene)
        index_hub.append(index)
# for task in pathway_task_list:
#     if task == 'CNA':
#         Term_CAN = dict_pathway['Term_CAN']
#         Term_CAN = extract_before_delimiters(Term_CAN)
#         Genes_CAN = dict_pathway['Genes_CAN']
#         CAN_dict = dict(zip(Term_CAN, Genes_CAN))
#         for p in pathway:
#             if p in CAN_dict:
#                 genes = CAN_dict[p]
#     elif task == 'Gene':
#         Term_Gene = dict_pathway['Term_Gene']
#         Term_Gene = extract_before_delimiters(Term_Gene)
#         Genes_Gene = dict_pathway['Genes_Gene']
#         Gene_dict = dict(zip(Term_Gene, Genes_Gene))
#         for p in pathway:
#             if p in Gene_dict:
#                 genes = Gene_dict[p]
#     elif task == 'Protein':
#         Term_Protein = dict_pathway['Term_Protein']
#         Term_Protein = extract_before_delimiters(Term_Protein)
#         Genes_Protein = dict_pathway['Genes_Protein']
#         Protein_dict = dict(zip(Term_Protein, Genes_Protein))
#         for p in pathway:
#             if p in Protein_dict:
#                 genes = Protein_dict[p]
#
# gene_list = genes.split(',')
# task_list = [task] * len(gene_list)
# gene_list = [gene.strip() for gene in gene_list]
# task_list = ['CNA','CNA']  # ['CNA','CNA','CNA','CNA','CNA','Protein','Protein','Protein','Protein','Gene','Gene','Gene']
# gene_list = ['CDKN2B-AS1','SFTA3']  # ['CDKN2B-AS1','CDKN2B','C9orf53','CLPTM1L','SFTA3','C-Raf_pS338','Rad51','14-3-3_epsilon','p53','TP53','TTN','MUC16']
##############################################################################################
thred = 85  # Journal version is 85
patch_thershold = 16
use_knn_graph = 'False'
pca_or_not = 'False'  # use pca in data-preprocessingxc
pca_right_now = 'True'  # use pca in this code
pca_feature_dimension = 64  # 64 edge 32
feature_dimension = 512
backbone = 'resnet18'

batch_size = 24  # 64
node_num = 1000
init_lr = 1e-3  # 30
wd = 1e-2  # 1e-2 # weight decay


###################################################################      Train and Test Function   ######################################################################
def visualize_gradients(model):
    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.view(-1).cpu().numpy())

    # 将所有梯度展开为一个数组
    all_grads = np.concatenate(grads)

    plt.hist(all_grads, bins=50)
    plt.title("Gradient Distribution")
    plt.xlabel("Gradient values")
    plt.ylabel("Frequency")
    plt.show()


def remove_padding(data_list):
    """
    移除无效的 padding 数据，判断依据为：
    - x 的所有值均为 0。
    - edge_index 的大小为 [2, 0]。
    - data_name 的值可能无意义（如过小）。

    :param data_list: 包含 Data 对象的列表
    :return: (cleaned_data, removed_indices)
        - cleaned_data: 移除无效数据后的列表
        - removed_indices: 被移除数据的索引
    """
    cleaned_data = []
    removed_indices = []

    for idx, data in enumerate(data_list):
        # 确保 data 是有效对象
        if data is None:
            removed_indices.append(idx)
            continue

        # 检查 x 是否全为 0
        is_x_zero = data.x is not None and torch.all(data.x == 0)

        # 检查 edge_index 是否为空
        is_edge_empty = data.edge_index is not None and data.edge_index.size(1) == 0

        # 检查 data_name 是否无意义
        is_data_name_invalid = data.data_name is not None and data.data_name[0].item() < 10

        # 如果符合所有移除条件
        if is_x_zero and is_edge_empty and is_data_name_invalid:
            removed_indices.append(idx)
        else:
            cleaned_data.append(data)

    return cleaned_data, removed_indices


# def test(loader):
#     model.eval()
#     output_list = []
#     select_index_list_all = []
#     added_node_features_all = []
#     self_graph_all = []
#     use_data_name = []
#
#
#     with torch.no_grad():  # 测试时不需要计算梯度
#         for data_list in loader:
#
#             # detail = [data.data_name for data in data_list]
#
#             detail = []
#             # 遍历 data_list 中的每个 data 对象
#             for data in data_list:
#                 # 将 data_name 转换为 Python 列表
#                 ascii_values = data.data_name.tolist()
#
#                 # 过滤掉无效的 ASCII 值
#                 valid_ascii_values = [value for value in ascii_values if 0 <= value < 0x110000]
#
#                 # 将有效的 ASCII 值转换为字符
#                 characters = [chr(value) for value in valid_ascii_values]
#
#                 # 将字符列表组合成字符串
#                 string = ''.join(characters)
#
#                 # 将字符串添加到 detail 列表中
#                 detail.append(string)
#
#             use_data_name.append(detail)
#             # 前向传播并获取预测结果和选定节点索引
#             output, select_index_list, added_node_features, self_graph = model(data_list)
#             data_list, removed_indices = remove_padding(data_list)
#             mask_output = torch.ones(output.size(0), dtype=torch.bool)
#             mask_output[removed_indices] = False
#             result_output = output[mask_output]
#
#             mask_select_index_list = torch.ones(select_index_list.size(0), dtype=torch.bool)
#             mask_select_index_list[removed_indices] = False
#             result_select_index_list = select_index_list[mask_select_index_list]
#
#             mask_added_node_features = torch.ones(added_node_features.size(1), dtype=torch.bool)
#             mask_added_node_features[removed_indices] = False
#             result_added_node_features = added_node_features[mask_added_node_features.unsqueeze(0)]
#
#
#             output_list.append(result_output)
#             select_index_list_all.append(result_select_index_list)
#             added_node_features_all.append(result_added_node_features)
#             self_graph_all.append(self_graph)
#
#
#     return output_list, select_index_list_all, added_node_features_all, self_graph_all, use_data_name

def test(loader, slide_name_order, dic_slide_patch, dataset_name,lambda_reg=0.01):
    model.eval()
    loss_all = 0
    correct = 0
    epoch_pred = []
    epoch_label = []
    epoch_score = []
    dic_slide_select_node = {}
    output_list = []
    select_index_list_all = []
    added_node_features_all = []
    self_graph_all = []
    use_data_name = []

    correct_sample = 0
    total_sample = 0
    accuracy_sample_sig_list = []

    with torch.no_grad():  # 测试时不需要计算梯度
        for data_list in loader:

            # detail = [data.data_name for data in data_list]

            detail = []
            # 遍历 data_list 中的每个 data 对象
            for data in data_list:
                # 将 data_name 转换为 Python 列表
                ascii_values = data.data_name.tolist()

                # 过滤掉无效的 ASCII 值
                valid_ascii_values = [value for value in ascii_values if 0 <= value < 0x110000]

                # 将有效的 ASCII 值转换为字符
                characters = [chr(value) for value in valid_ascii_values]

                # 将字符列表组合成字符串
                string = ''.join(characters)

                # 将字符串添加到 detail 列表中
                detail.append(string)

            use_data_name.append(detail)
            # 前向传播并获取预测结果和选定节点索引
            output, select_index_list, added_node_features, self_graph = model(data_list)
            data_list, removed_indices = remove_padding(data_list)
            mask = torch.ones(output.size(0), dtype=torch.bool)
            mask[removed_indices] = False
            result_output = output[mask]
            y = torch.cat([data.y.unsqueeze(0) for data in data_list]).to(output.device)
            y = y.view(result_output.size()).float()


            mask_select_index_list = torch.ones(select_index_list.size(0), dtype=torch.bool)
            mask_select_index_list[removed_indices] = False
            result_select_index_list = select_index_list[mask_select_index_list]

            mask_added_node_features = torch.ones(added_node_features.size(1), dtype=torch.bool)
            mask_added_node_features[removed_indices] = False
            result_added_node_features = added_node_features[mask_added_node_features.unsqueeze(0)]


            output_list.append(result_output)
            select_index_list_all.append(result_select_index_list)
            added_node_features_all.append(result_added_node_features)
            self_graph_all.append(self_graph)

            # 计算损失
            # 计算正则化项（L2 范数）
            l2_reg = lambda_reg * torch.norm(self_graph.edge_attr, p=2)

            loss = loss_fn(result_output, y)
            loss += l2_reg  # 添加正则化项
            loss_all += loss.item()

            # 二值化预测结果
            probs = torch.sigmoid(result_output)
            thresholds = torch.full(probs.shape, 0.5, device=probs.device)

            pred = (probs > thresholds).float()  # 使用阈值0.5二值化

            # 样本级别的准确预测数
            correct_list = pred.eq(y)
            correct_list = [sublist.all().item() for sublist in correct_list]
            correct_tensor = torch.tensor(correct_list, dtype=torch.int32, device=result_output.device)
            correct += correct_tensor.sum().item()

            # 保存预测、标签和分数
            epoch_score.extend(result_output.cpu().numpy())
            epoch_pred.extend(pred.cpu().numpy())
            epoch_label.extend(y.cpu().numpy())

            # 处理选定的节点索引
            select_index_list = select_index_list.cpu().tolist()
            b = [[int(j) for j in i] for i in select_index_list]
            for graph, index_list in enumerate(b, 0):
                # print(graph)
                # print(index_list)
                # slide_name = slide_name_order[detail[graph]]
                if detail[graph] in slide_name_order:
                    slide_name = detail[graph]
                # slide_name = detail[graph]
                node_list = dic_slide_patch[slide_name]
                select_node = [node_list[i] for i in index_list]
                dic_slide_select_node[slide_name] = select_node

    # 将预测值和真实标签转换为numpy数组
    epoch_pred = np.array(epoch_pred)
    epoch_label = np.array(epoch_label)
    epoch_score = np.array(epoch_score)

    # 计算每个标签的独立准确率
    label_acc = []
    for i in range(epoch_label.shape[1]):
        label_acc.append(accuracy_score(epoch_label[:, i], epoch_pred[:, i]))

    # 计算每个标签的独立 AUC
    label_auc = []
    for i in range(epoch_label.shape[1]):
        try:
            auc = roc_auc_score(epoch_label[:, i], epoch_score[:, i])
        except ValueError:
            auc = None  # 如果只有一个类别，AUC 不适用
        label_auc.append(auc)

    # 样本级别准确率（每个样本的所有标签必须完全匹配）
    sample_acc = accuracy_score(epoch_label, epoch_pred, normalize=True)

    # 微平均准确率（针对所有样本和标签，整体计算准确率）
    micro_acc = precision_score(epoch_label, epoch_pred, average='micro')

    # 宏平均准确率（先计算每个标签的准确率，再取平均）
    macro_acc = precision_score(epoch_label, epoch_pred, average='macro')

    # Hamming Loss（标签错误率）
    hamming = hamming_loss(epoch_label, epoch_pred)

    # 返回结果，包括各种评估指标
    return output_list, select_index_list_all, added_node_features_all, self_graph_all, use_data_name, loss_all / len(loader),sample_acc, micro_acc, macro_acc, hamming, epoch_pred, epoch_score, epoch_label, dic_slide_select_node, label_acc, label_auc


# model_path = "/mnt/data16t/yuy/pathway/patch/only_change/SAGE/model/task_Gene/gene_hsa04151/model/flod_4/epoch_71.pth"
# num_classes = len(gene_list)
# edge_values = torch.full((num_classes, 1), 1 / num_classes, dtype=torch.float32)
# new_value = 2 / num_classes  # 你想要的新值
# new_values = torch.full((len(index_hub), 1), new_value, dtype=torch.float32)
# for index, value in zip(index_hub, new_values):
#     edge_values[index] = value
# model = Net(num_classes, 24, edge_values)  # 初始化模型
# print("Let's use", torch.cuda.device_count(), "GPUs!")
# model = DataParallel(model, device_ids=[0])  # 如果使用多GPU，调整设备ID
# model.load_state_dict(torch.load(model_path))
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# model.to(device)

###################################################################      Load TCGA-COAD data    ######################################################################

start = time.time()
if pca_or_not == 'False':

    feature_path1 = '/mnt/data16t/yuy/output_lung/luad_5_flod/patch/all_Journal_coad_feature{}_{}_v1.json'.format(
        feature_dimension, backbone)
    feature_name_path1 = '/mnt/data16t/yuy/output_lung/luad_5_flod/patch/all_Journal_coad_feature{}_{}_name_v1.json'.format(
        feature_dimension, backbone)
    adj_path1 = '/mnt/data16t/yuy/output_lung/luad_5_flod/patch/all_Journal_coad_feature{}_{}_adj_{}_v1.json'.format(
        feature_dimension, backbone, thred)

    feature_path2 = '/mnt/data16t/yuy/output_lung/luad_5_flod/patch/all_Journal_coad_feature{}_{}_v2.json'.format(
        feature_dimension, backbone)
    feature_name_path2 = '/mnt/data16t/yuy/output_lung/luad_5_flod/patch/all_Journal_coad_feature{}_{}_name_v2.json'.format(
        feature_dimension, backbone)
    adj_path2 = '/mnt/data16t/yuy/output_lung/luad_5_flod/patch/all_Journal_coad_feature{}_{}_adj_{}_v2.json'.format(
        feature_dimension, backbone, thred)

    feature_path3 = '/mnt/data16t/yuy/output_lung/luad_5_flod/patch/all_Journal_coad_feature{}_{}_v3.json'.format(
        feature_dimension, backbone)
    feature_name_path3 = '/mnt/data16t/yuy/output_lung/luad_5_flod/patch/all_Journal_coad_feature{}_{}_name_v3.json'.format(
        feature_dimension, backbone)
    adj_path3 = '/mnt/data16t/yuy/output_lung/luad_5_flod/patch/all_Journal_coad_feature{}_{}_adj_{}_v3.json'.format(
        feature_dimension, backbone, thred)

    feature_path4 = '/mnt/data16t/yuy/output_lung/luad_5_flod/patch/all_Journal_coad_feature{}_{}_v4.json'.format(
        feature_dimension, backbone)
    feature_name_path4 = '/mnt/data16t/yuy/output_lung/luad_5_flod/patch/all_Journal_coad_feature{}_{}_name_v4.json'.format(
        feature_dimension, backbone)
    adj_path4 = '/mnt/data16t/yuy/output_lung/luad_5_flod/patch/all_Journal_coad_feature{}_{}_adj_{}_v4.json'.format(
        feature_dimension, backbone, thred)

    feature_path5 = '/mnt/data16t/yuy/output_lung/luad_5_flod/patch/all_Journal_coad_feature{}_{}_v5.json'.format(
        feature_dimension, backbone)
    feature_name_path5 = '/mnt/data16t/yuy/output_lung/luad_5_flod/patch/all_Journal_coad_feature{}_{}_name_v5.json'.format(
        feature_dimension, backbone)
    adj_path5 = '/mnt/data16t/yuy/output_lung/luad_5_flod/patch/all_Journal_coad_feature{}_{}_adj_{}_v5.json'.format(
        feature_dimension, backbone, thred)

else:
    feature_path1 = '/mnt/data16t/yuy/output_lung/luad_5_flod/patch/all_Journal_coad_feature{}_{}_pca_v1.json'.format(
        feature_dimension, backbone)
    feature_name_path1 = '/mnt/data16t/yuy/output_lung/luad_5_flod/patch/all_Journal_coad_feature{}_{}_pca_name_v1.json'.format(
        feature_dimension, backbone)
    adj_path1 = '/mnt/data16t/yuy/output_lung/luad_5_flod/patch/all_Journal_coad_feature{}_{}_pca_adj_v1.json'.format(
        feature_dimension, backbone)

    feature_path2 = '/mnt/data16t/yuy/output_lung/luad_5_flod/patch/all_Journal_coad_feature{}_{}_pca_v2.json'.format(
        feature_dimension, backbone)
    feature_name_path2 = '/mnt/data16t/yuy/output_lung/luad_5_flod/patch/all_Journal_coad_feature{}_{}_pca_name_v2.json'.format(
        feature_dimension, backbone)
    adj_path2 = '/mnt/data16t/yuy/output_lung/luad_5_flod/patch/all_Journal_coad_feature{}_{}_pca_adj_v2.json'.format(
        feature_dimension, backbone)

    feature_path3 = '/mnt/data16t/yuy/output_lung/luad_5_flod/patch/all_Journal_coad_feature{}_{}_pca_v3.json'.format(
        feature_dimension, backbone)
    feature_name_path3 = '/mnt/data16t/yuy/output_lung/luad_5_flod/patch/all_Journal_coad_feature{}_{}_pca_name_v3.json'.format(
        feature_dimension, backbone)
    adj_path3 = '/mnt/data16t/yuy/output_lung/luad_5_flod/patch/all_Journal_coad_feature{}_{}_pca_adj_v3.json'.format(
        feature_dimension, backbone)

    feature_path4 = '/mnt/data16t/yuy/output_lung/luad_5_flod/patch/all_Journal_coad_feature{}_{}_pca_v4.json'.format(
        feature_dimension, backbone)
    feature_name_path4 = '/mnt/data16t/yuy/output_lung/luad_5_flod/patch/all_Journal_coad_feature{}_{}_pca_name_v4.json'.format(
        feature_dimension, backbone)
    adj_path4 = '/mnt/data16t/yuy/output_lung/luad_5_flod/patch/all_Journal_coad_feature{}_{}_pca_adj_v4.json'.format(
        feature_dimension, backbone)

    feature_path5 = '/mnt/data16t/yuy/output_lung/luad_5_flod/patch/all_Journal_coad_feature{}_{}_pca_v5.json'.format(
        feature_dimension, backbone)
    feature_name_path5 = '/mnt/data16t/yuy/output_lung/luad_5_flod/patch/all_Journal_coad_feature{}_{}_pca_name_v5.json'.format(
        feature_dimension, backbone)
    adj_path5 = '/mnt/data16t/yuy/output_lung/luad_5_flod/patch/all_Journal_coad_feature{}_{}_pca_adj_v5.json'.format(
        feature_dimension, backbone)

print(adj_path5)
dic_name_data1, data_name, slide_name_order1, dic_slide_patch1 = generate_dataset(pca_feature_dimension, feature_path1,
                                                                                  feature_name_path1, adj_path1,
                                                                                  use_knn_graph, pca_right_now)
print('finish load dataset1')
dic_name_data2, data_name, slide_name_order2, dic_slide_patch2 = generate_dataset(pca_feature_dimension, feature_path2,
                                                                                  feature_name_path2, adj_path2,
                                                                                  use_knn_graph, pca_right_now)
print('finish load dataset2')
dic_name_data3, data_name, slide_name_order3, dic_slide_patch3 = generate_dataset(pca_feature_dimension, feature_path3,
                                                                                  feature_name_path3, adj_path3,
                                                                                  use_knn_graph, pca_right_now)
print('finish load dataset3')
dic_name_data4, data_name, slide_name_order4, dic_slide_patch4 = generate_dataset(pca_feature_dimension, feature_path4,
                                                                                  feature_name_path4, adj_path4,
                                                                                  use_knn_graph, pca_right_now)
print('finish load dataset4')
dic_name_data5, data_name, slide_name_order5, dic_slide_patch5 = generate_dataset(pca_feature_dimension, feature_path5,
                                                                                  feature_name_path5, adj_path5,
                                                                                  use_knn_graph, pca_right_now)
print('finish load dataset5')
#
slide_name_order_list = [slide_name_order1, slide_name_order2, slide_name_order3, slide_name_order4, slide_name_order5]
dic_slide_patch_list = [dic_slide_patch1, dic_slide_patch2, dic_slide_patch3, dic_slide_patch4, dic_slide_patch5]
#
# slide_name_order_list = [slide_name_order1]
# dic_slide_patch_list = [dic_slide_patch1]
slide_name_order_list = [list(chain.from_iterable(slide_name_order_list))]
# dic_slide_patch_list = [{**dic_slide_patch1, **dic_slide_patch2, **dic_slide_patch3, **dic_slide_patch4, **dic_slide_patch5}]

# dic_slide_patch_list = [dic_slide_patch1,dic_slide_patch2]
# dic_slide_patch_list = [{**dic_slide_patch1, **dic_slide_patch2}]
###################################################################      Training start here    ######################################################################

result_gene = {}  # {'gene' : {1:{1:[acc,auc,....], 2:[...]}, 2:{...}}}
gene_ensemble_result = {}
# start = time.time()

save_train_num = []

output_all = []
select_index_all = []
all_added_node_features = []
all_self_graph = []
test_data_patch_lists = []

for gene, task in zip(pathway, pathway_task_list):
    if task == 'MSI':
        label_path = 'MSI_MSS_coad.json'
        with open(label_path) as f:
            dic_id_label = json.load(f)

    else:
        if task == 'CNA':
            label_path = '/mnt/data16t/yuy/data/top20_LUAD_CNA.csv'
        else:
            label_path = '/mnt/data16t/yuy/data/top20_LUAD.csv'
        dataset = pd.read_csv(label_path, delimiter=",")
        df = pd.DataFrame(dataset)

        # load label
        dic_id_label = {}
        dic_id_label_list = {}
        name = df['id'].values.tolist()
        # label = df[gene].values.tolist()
        label_list = df[gene_list].values.tolist()


        false_label_slide = []
        for index, n, in enumerate(name, 0):
            if any(label_val == 2 for label_val in label_list[index]):
                print(n)
                print(label_list[index])
                false_label_slide.append(n)
                continue
            else:
                # dic_id_label[n] = label[index]
                dic_id_label_list[n] = label_list[index]


    ##############################################################################################
    def add_labels(dic_name_data, dic_id_label):
        """
        根据dic_id_label为dic_name_data中的每个样本添加标签，并按标签值将样本分为正负样本。
        确保正负样本列表中的样本没有重复。
        """

        label_count = len(next(iter(dic_id_label.values())))  # 获取标签数量
        pos_name_data = [[] for _ in range(label_count)]  # 存储每个标签对应的正样本
        neg_name_data = [[] for _ in range(label_count)]  # 存储每个标签对应的负样本
        name_neg = [[] for _ in range(label_count)]  # 用列表保存负样本数据
        name_pos = [[] for _ in range(label_count)]  # 用列表保存正样本数据
        n = 1  # 每个标签的最少的正负样本数量

        # 1. 首次分配正负样本
        for key, data in dic_name_data.items():
            id = key[0:12]  # 获取样本的ID（前12个字符）

            if id in dic_id_label.keys():
                labels = dic_id_label[id]
                y = torch.from_numpy(np.array(labels))  # 将标签转换为torch tensor

                # 计算每个标签的正负样本数量
                positive_counts = [0] * len(labels)  # 初始化每个标签的正样本计数
                negative_counts = [0] * len(labels)  # 初始化每个标签的负样本计数

                # 遍历所有样本，统计每个标签的正负样本数量
                for sample_id, labels_list in dic_id_label.items():
                    for i, label in enumerate(labels_list):
                        if label == 1:
                            positive_counts[i] += 1  # 正样本
                        else:
                            negative_counts[i] += 1  # 负样本

                # 创建一个列表，将正样本和负样本按标签索引分开处理
                label_priority = []

                # 按照每个标签的正样本和负样本数量分别打包，并标明类型
                for i in range(len(labels)):
                    # 将正样本加入
                    label_priority.append((positive_counts[i], i, 'pos'))  # ('正样本数量', '标签索引', '类型')
                    # 将负样本加入
                    label_priority.append((negative_counts[i], i, 'neg'))  # ('负样本数量', '标签索引', '类型')

                # 按照正负样本数量升序排序
                label_priority.sort(key=lambda x: x[0])  # 按正负样本数量排序

                # 输出排序后的标签顺序
                sorted_labels_order = [x for x in label_priority]  # 排序后的标签顺序

                # 打印排序后的顺序
                # print("排序后的标签顺序（按数量升序）：")
                # for count, index, sample_type in sorted_labels_order:
                #     print(f"标签{index} - 类型: {sample_type} - 数量: {count}")

                # 根据排序后的标签顺序来生成shifted_labels
                sort_list = []
                for i, label in enumerate(labels):
                    if label == 1:
                        sample_type = 'pos'
                        # 过滤出所有 sample_type == 'pos' 的元素
                        pos_elements = [(index, num, count, sample_type) for index, (num, count, sample_type) in
                                        enumerate(label_priority) if sample_type == 'pos']

                        count = i  # 查找正样本数量为 i 的元素
                        found_elements = [pos for pos in pos_elements if pos[2] == count]
                        for element in found_elements:
                            index, num, count, sample_type = element
                            need = (num, count, sample_type)
                            sorted_position = sorted_labels_order.index(need)
                        sort_list.append((sorted_position, count, sample_type))
                    else:
                        sample_type = 'neg'
                        # 过滤出所有 sample_type == 'neg' 的元素
                        neg_elements = [(index, num, count, sample_type) for index, (num, count, sample_type) in
                                        enumerate(label_priority) if sample_type == 'neg']
                        count = i  # 查找负样本数量为 i 的元素
                        found_elements = [neg for neg in neg_elements if neg[2] == count]
                        for element in found_elements:
                            index, num, count, sample_type = element
                            need = (num, count, sample_type)
                            sorted_neg = sorted_labels_order.index(need)
                        sort_list.append((sorted_neg, count, sample_type))

                sort_list.sort(key=lambda x: x[0])

                # 根据 sort_list 的顺序调整 labels
                # 提取排序后的索引
                sorted_indices = [item[1] for item in sort_list]  # 获取按照 sorted_neg 排序后的索引列表

                # 更新 labels 按照 sorted_indices 的顺序
                sorted_labels = [labels[i] for i in sorted_indices]

                # 使用 enumerate 对shifted_labels进行遍历
                for label, sort_ele in zip(sorted_labels, sort_list):
                    sort_index, sort_count, sort_sample_type = sort_ele
                    if label == 1:  # 如果标签为1，添加到正样本列表
                        if key not in [item for sublist in name_neg for item in sublist]:  # 确保不重复
                            data.y = y
                            if key not in [item for sublist in name_pos for item in sublist]:
                                name_pos[sort_count].append(key)
                            dic_name_data[key] = data  # 更新dic_name_data中的数据
                    else:  # 如果标签为0，添加到负样本列表
                        if key not in [item for sublist in name_pos for item in sublist]:  # 确保不重复
                            data.y = y
                            if key not in [item for sublist in name_neg for item in sublist]:
                                name_neg[sort_count].append(key)
                            dic_name_data[key] = data  # 更新dic_name_data中的数据

        n = 5  # 每个标签至少需要 n 个样本

        # 2. 重新检查，确保每个标签的正负样本列表不为空
        for i in range(label_count):
            # 如果正样本列表为空
            if len(name_pos[i]) < 1:
                # 重新检查当前循环中的样本，找出一个填充
                for key, data in dic_name_data.items():
                    id = key[0:12]  # 获取样本的ID
                    if id in dic_id_label.keys():
                        labels = dic_id_label[id]  # 获取该样本的标签
                        y = torch.from_numpy(np.array(labels))  # 转为PyTorch tensor

                        if key not in name_pos[i] and key not in name_neg[i]:  # 确保样本未被添加
                            # 填充正样本
                            if labels[i] == 1 and len(name_pos[i]) < 1:  # 找到正样本
                                data.y = y
                                if key not in [item for sublist in name_pos for item in sublist]:
                                    name_pos[i].append(key)
                                dic_name_data[key] = data  # 更新dic_name_data中的数据
                                break  # 只添加一个样本

                # 再次检查，如果正样本仍不足，从负样本中迁移
                if len(name_pos[i]) < n and len(name_neg[i]) > 0:
                    for j in range(label_count):  # 从其他标签迁移
                        if j != i and len(name_neg[j]) > 0:
                            # 从负样本中删除一个样本，加入到正样本中
                            moved_sample = name_neg[j].pop()
                            name_pos[i].append(moved_sample)
                            # print(f"Moved sample {moved_sample} from negative to positive for label {i}.")
                            if len(name_pos[i]) >= n:  # 一旦满足条件，停止迁移
                                break

            elif len(name_neg[i]) < 1:
                # 重新检查当前循环中的样本，找出一个填充
                for key, data in dic_name_data.items():
                    id = key[0:12]  # 获取样本的ID
                    if id in dic_id_label.keys():
                        labels = dic_id_label[id]  # 获取该样本的标签
                        y = torch.from_numpy(np.array(labels))  # 转为PyTorch tensor

                        if key not in name_pos[i] and key not in name_neg[i]:  # 确保样本未被添加
                            # 填充负样本
                            if labels[i] == 0 and len(name_neg[i]) < 1:  # 找到负样本
                                data.y = y
                                if key not in [item for sublist in name_neg for item in sublist]:
                                    name_neg[i].append(key)
                                dic_name_data[key] = data
                                break  # 只添加一个样本

                # 再次检查，如果负样本仍不足，从正样本中迁移
                if len(name_neg[i]) < n and len(name_pos[i]) > 0:
                    for j in range(label_count):  # 从其他标签迁移
                        if j != i and len(name_pos[j]) > 0:
                            # 从正样本中删除一个样本，加入到负样本中
                            moved_sample = name_pos[j].pop()
                            name_neg[i].append(moved_sample)
                            # print(f"Moved sample {moved_sample} from positive to negative for label {i}.")
                            if len(name_neg[i]) >= n:  # 一旦满足条件，停止迁移
                                break

        return name_neg, name_pos, dic_name_data


    name_negs1, name_poss1, dic_name_datas1 = add_labels(dic_name_data1, dic_id_label_list)
    name_negs2, name_poss2, dic_name_datas2 = add_labels(dic_name_data2, dic_id_label_list)
    name_negs3, name_poss3, dic_name_datas3 = add_labels(dic_name_data3, dic_id_label_list)
    name_negs4, name_poss4, dic_name_datas4 = add_labels(dic_name_data4, dic_id_label_list)
    name_negs5, name_poss5, dic_name_datas5 = add_labels(dic_name_data5, dic_id_label_list)

    dic_name_data_list = [dic_name_data1, dic_name_data2, dic_name_data3, dic_name_data4, dic_name_data5]
    # dic_name_data_list = [dic_name_data1]
    # dic_name_data = {**dic_name_data1, **dic_name_data2, **dic_name_data3, **dic_name_data4, **dic_name_data5}

    # dic_name_data_list = [dic_name_data1, dic_name_data2]
    # dic_name_data_list = {**dic_name_datas1, **dic_name_datas2}

    slide_name_order = []
    dic_slide_patch = {}
    for slide_name_order1, dic_slide_patch1 in zip(slide_name_order_list, dic_slide_patch_list):
        slide_name_order.extend(slide_name_order1)
        dic_slide_patch.update(dic_slide_patch1)


    for dic_name_data_now, dic_slide_patch_now in zip(dic_name_data_list, dic_slide_patch_list):
        test_data_patch_list = []
        # test_dataset = dic_name_data_now
        test_name = list(dic_name_data_now.keys())
        # test_slide_dic = dic_slide_patch_now[test_name]

        def remove_y_from_data(data_dict):
            """
            从字典中移除 Data 对象的 y 属性。

            Args:
                data_dict (dict): 键为数据名称，值为 torch_geometric.data.Data 对象。

            Returns:
                dict: 移除 y 属性后的 Data 对象字典。
            """
            cleaned_data_dict = {}
            for key, data in data_dict.items():
                # 如果 Data 对象存在 y 属性，删除
                if hasattr(data, 'y'):
                    del data.y
                cleaned_data_dict[key] = data
            return cleaned_data_dict


        def remove_data_without_y(data_dict):
            """
            移除没有有效 y 属性的 Data 对象。
            """
            cleaned_data_dict = {}
            for key, data in data_dict.items():
                # 检查 y 属性是否存在且不是 None
                if hasattr(data, 'y') and data.y is not None:
                    cleaned_data_dict[key] = data
            return cleaned_data_dict



        # test_dataset = remove_y_from_data(dic_name_data_now)
        test_dataset = remove_data_without_y(dic_name_data_now)

        def pad_last_batch(data_dict, batch_size):
            # 将字典中的 Data 对象转换为列表
            data_list = list(data_dict.values())
            num_data = len(data_list)

            # 计算需要填充的数量
            padding_needed = (batch_size - (num_data % batch_size)) % batch_size

            # 创建填充的 Data 对象，确保属性与原始数据一致
            padded_data_list = data_list + [
                Data(
                    x=torch.zeros(1, data_list[0].x.size(1)),
                    edge_index=torch.empty((2, 0), dtype=torch.long),
                    data_name=torch.tensor([-1]),  # 用 -1 或其他值表示填充数据
                    y=torch.tensor([-1])  # 假设 y 是标签属性
                ) for _ in range(padding_needed)
            ]

            return padded_data_list


        test_dataset = pad_last_batch(test_dataset, batch_size)
        test_loader = DataListLoader(test_dataset, batch_size=batch_size, shuffle=False)
        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        num_classes = len(gene_list)
        edge_values = torch.full((num_classes, 1), 1 / num_classes, dtype=torch.float32)
        new_value = 2 / num_classes  # 你想要的新值
        new_values = torch.full((len(index_hub), 1), new_value, dtype=torch.float32)
        for index, value in zip(index_hub, new_values):
            edge_values[index] = value
        model = Net(num_classes,24, edge_values)  # 初始化模型
        # 加载预训练模型参数
        model_path = "/mnt/data16t/yuy/pathway/patch/only_change/GIN_change/model_hsa04151/task_Gene/gene_hsa04151/model/flod_5/epoch_97.pth"

        # 配置多GPU和设备
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
        model = DataParallel(model, device_ids=[2])  # 如果使用多GPU，调整设备ID
        model.load_state_dict(torch.load(model_path))

        model.to(device)

        # 测试模式
        model.eval()  # 切换为评估模式，关闭Dropout和BatchNorm

        # 定义损失函数和优化器（测试时不需要优化器）
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
        # loss_fn = nn.BCEWithLogitsLoss()

        # output, select_index_list, added_node_features, self_graph, use_data_names = test(test_loader)

        output, select_index_list, added_node_features, self_graph, use_data_names, test_loss, test_acc, test_micro_acc, test_macro_acc, test_hamming_loss, pred_label, pred_score, epoch_label, epoch_slide_select_patch, test_label_acc, test_label_auc = test(
            test_loader, slide_name_order, dic_slide_patch_now, dataset_name='testing')

        # 计算 TP, TN, FP, FN
        TP = np.sum((epoch_label == 1) & (pred_label == 1), axis=0)  # 每列对应一个标签
        TN = np.sum((epoch_label == 0) & (pred_label == 0), axis=0)
        FP = np.sum((epoch_label == 0) & (pred_label == 1), axis=0)
        FN = np.sum((epoch_label == 1) & (pred_label == 0), axis=0)

        # 总 TP, TN, FP, FN
        total_TP = np.sum(TP)
        total_TN = np.sum(TN)
        total_FP = np.sum(FP)
        total_FN = np.sum(FN)

        # 微平均 Precision, Recall, F1
        micro_precision = total_TP / (total_TP + total_FP)
        micro_recall = total_TP / (total_TP + total_FN)
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)

        # 宏平均 Precision, Recall, F1
        precision_per_label = TP / (TP + FP + 1e-8)  # 防止分母为0
        recall_per_label = TP / (TP + FN + 1e-8)
        f1_per_label = 2 * precision_per_label * recall_per_label / (precision_per_label + recall_per_label + 1e-8)
        macro_f1 = np.mean(f1_per_label)

        # 计算 Precision, Recall, F1 per label
        precision_per_label = TP / (TP + FP + 1e-8)  # 防止分母为 0
        recall_per_label = TP / (TP + FN + 1e-8)
        f1_per_label = 2 * precision_per_label * recall_per_label / (precision_per_label + recall_per_label + 1e-8)

        # 将每个标签的 F1 分数存储为列表
        f1_scores_list = f1_per_label.tolist()

        y = epoch_label
        y_score = np.array(pred_score)
        # Calculate ROC curve and AUC for each class
        # 计算宏平均的 ROC AUC 分数
        # roc_auc_macro = roc_auc_score(y, y_score, average='macro')
        # 计算微平均的 ROC AUC 分数
        # roc_auc_micro = roc_auc_score(y, y_score, average='micro')
        # fpr, tpr, thresholds = metrics.roc_curve(y, y_score, pos_label=1)
        # test_auc = metrics.auc(fpr, tpr)
        print(
            f'Test Loss: {test_loss:.7f} | Test Acc: {test_acc} | '
            f'Test Macro-Averaging Acc: {test_micro_acc} | Test Micro-Averaging Acc: {test_macro_acc} |\n'
            # f'Test Macro-Averaging Auc: {roc_auc_macro} | Test Micro-Averaging Auc: {roc_auc_micro} |\n'
            f'Test Hamming Loss: {test_hamming_loss:.7f}\n'
        )
        print(f'Gene_List: {gene_list}')
        print(f'test_label_acc: {test_label_acc}')
        print(f'test_label_auc: {test_label_auc}')
        print(f'TP: {TP}')
        print(f'TN: {TN}')
        print(f'FP: {FP}')
        print(f'FN: {FN}')
        print(f'Total TP:{total_TP}')
        print(f'Total TN: {total_TN}')
        print(f'Total FP: {total_FP}')
        print(f'Total FN: {total_FN}')
        print(f'Micro F1: {micro_f1}')
        print(f'Macro F1: {macro_f1}')
        print(f'F1 Scores per label: {f1_scores_list}')

        print()

        output_all.append(output)
        cleaned_list = [[item for item in sublist if item] for sublist in use_data_names]
        for select_indexs, data_name_list_now in zip(select_index_list, cleaned_list):
            select_test_data_patch = []
            for select_index, data_name_now in zip(select_indexs, data_name_list_now):
                slide_patch_name_now = dic_slide_patch_now[data_name_now]
                test_data_patch = []
                for index in select_index:
                    slide_patch_name = slide_patch_name_now[index.int().item()]
                    test_data_patch.append(slide_patch_name)
                select_test_data_patch.append(test_data_patch)
            test_data_patch_list.append(select_test_data_patch)
        test_data_patch_lists.append(test_data_patch_list)
        select_index_all.append(select_index_list)
        all_added_node_features.append(added_node_features)
        all_self_graph.append(self_graph)
        print()

data_save_path = '/mnt/data16t/yuy/pathway/patch/only_change/GIN_change/hsa04151_ke_output'
if not os.path.exists(data_save_path):
    os.makedirs(data_save_path)

torch.save(test_data_patch_lists, '/mnt/data16t/yuy/pathway/patch/only_change/GIN_change/hsa04151_ke_output/test_data_patch.pth')
torch.save(output_all, '/mnt/data16t/yuy/pathway/patch/only_change/GIN_change/hsa04151_ke_output/output_all.pth')
torch.save(select_index_all, '/mnt/data16t/yuy/pathway/patch/only_change/GIN_change/hsa04151_ke_output/select_index_all.pth')
torch.save(all_added_node_features, '/mnt/data16t/yuy/pathway/patch/only_change/GIN_change/hsa04151_ke_output/all_added_node_features.pth')
torch.save(all_self_graph, '/mnt/data16t/yuy/pathway/patch/only_change/GIN_change/hsa04151_ke_output/all_self_graph.pth')

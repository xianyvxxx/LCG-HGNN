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

from torch.utils.data import DataLoader, WeightedRandomSampler
from torch_geometric.data import Data
from torch_geometric.loader import DataListLoader
from torch_geometric.nn import SAGEConv, GINConv, DenseGCNConv, GCNConv, GATConv, EdgeConv
# from torch_geometric.utils import dense_to_sparse, dropout_adj,true_positive, true_negative, false_positive, false_negative, precision, recall, f1_score
from torch_geometric.nn import knn_graph, dense_diff_pool, global_add_pool, global_mean_pool, DataParallel
from torch_geometric.nn import JumpingKnowledge as jp
from torch_geometric.utils import to_dense_batch

from itertools import chain

from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, accuracy_score, hamming_loss, roc_auc_score, f1_score, recall_score



##############################################################################################
# Customized
# from functions.model_architecture import Net, global_sort_pool
from multi_just_GAT import Net, global_sort_pool
from functions.utils import generate_dataset, oversampling, majority_vote
##############################################################################################
import csv

import torch
import time
from datetime import datetime

# 保存检查点
def save_checkpoint(state, filename='/mnt/data16t/yuy/pathway/patch/only_change/GAT/checkpoint_RHSA8948216.pth.tar'):
    """
    保存训练的检查点，包含当前的 fold、epoch、模型状态和优化器状态。
    """
    torch.save(state, filename)

# 加载检查点
def load_checkpoint(model, optimizer, filename='/mnt/data16t/yuy/pathway/patch/only_change/GAT/checkpoint_RHSA8948216.pth.tar'):
    """
    加载之前保存的检查点，恢复训练状态。
    """
    start_fold = 0
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_fold = checkpoint['fold']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (fold {}) (epoch {})"
              .format(filename, start_fold, start_epoch))
    else:
        print("=> no checkpoint found at '{}'".format(filename))
    return start_fold, start_epoch

# 保存 fold 索引
def save_fold_indices(fold_indices, samples, filename='/mnt/data16t/yuy/pathway/patch/only_change/GAT/fold_indices_RHSA8948216.json'):
    """
    保存每个 fold 的训练集和验证集的样本索引及对应的具体数据地址。
    """
    indices_to_save = []
    for train_idx, val_idx in fold_indices:
        train_samples = [samples[i] for i in train_idx]
        val_samples = [samples[i] for i in val_idx]
        indices_to_save.append({
            'train_idx': train_idx.tolist(),
            'val_idx': val_idx.tolist(),
            'train_samples': train_samples,
            'val_samples': val_samples
        })
    with open(filename, 'w') as f:
        json.dump(indices_to_save, f)


def load_fold_indices(samples, filename='/mnt/data16t/yuy/pathway/patch/only_change/GAT/fold_indices_RHSA8948216.json'):
    """
    加载之前保存的 fold 索引及对应的具体数据地址。
    """
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            indices = json.load(f)
        return [
            (np.array(item['train_idx']), np.array(item['val_idx']))
            for item in indices
        ]
    return None

def check_cuda_memory(device_id=2, threshold_mb=40000):
    """
    检查指定 CUDA 设备的空闲内存是否超过给定阈值。

    参数:
        device_id (int): 要检查的 CUDA 设备 ID，默认为 2
        threshold_mb (int): 空闲内存阈值（MB），默认为 40000 MB

    返回:
        bool: 如果空闲内存超过阈值返回 True，否则返回 False
    """
    if not torch.cuda.is_available():
        print("CUDA 不可用")
        return False

    # 查询 CUDA 内存信息
    memory_info = torch.cuda.mem_get_info(device=device_id)
    free_memory_bytes, total_memory_bytes = memory_info
    free_memory_mb = free_memory_bytes / (1024 ** 2)

    print(f"[{datetime.now()}] 设备 {device_id} 的总内存: {total_memory_bytes / (1024 ** 2):.2f} MB")
    print(f"[{datetime.now()}] 设备 {device_id} 的空闲内存: {free_memory_mb:.2f} MB")

    return free_memory_mb > threshold_mb


def run_next_step():
    """
    执行下一步的操作。
    """
    print("开始执行下一步操作...")
    # 在这里放置你想要执行的代码逻辑
    pass


def wait_for_memory_with_frequent_checks(device_id=2, threshold_mb=40000, initial_interval=1800, frequent_interval=60,
                                         consecutive_passes=5, final_wait=1800):
    """
    初始以较大间隔检查 CUDA 内存，当满足条件时转为高频检查，连续多次通过后再等待一段时间执行下一步。

    参数:
        device_id (int): 要检查的 CUDA 设备 ID，默认为 2
        threshold_mb (int): 空闲内存阈值（MB），默认为 40000 MB
        initial_interval (int): 初始检查间隔时间（秒），默认为 1800 秒（半小时）
        frequent_interval (int): 高频检查间隔时间（秒），默认为 60 秒（一分钟）
        consecutive_passes (int): 需要连续通过的次数，默认为 5 次
        final_wait (int): 连续通过后最终等待的时间（秒），默认为 1800 秒（半小时）
    """
    consecutive_count = 0

    while True:
        if check_cuda_memory(device_id, threshold_mb):
            print(f"[{datetime.now()}] 空闲内存超过 {threshold_mb} MB，开始高频检测")

            for _ in range(consecutive_passes):
                if check_cuda_memory(device_id, threshold_mb):
                    consecutive_count += 1
                    print(f"[{datetime.now()}] 高频检测通过 ({consecutive_count}/{consecutive_passes})")
                    time.sleep(frequent_interval)
                else:
                    print(f"[{datetime.now()}] 高频检测未通过，重置计数器")
                    consecutive_count = 0
                    break

            if consecutive_count == consecutive_passes:
                print(
                    f"[{datetime.now()}] 连续 {consecutive_passes} 次高频检测通过，等待 {final_wait / 60} 分钟后开始执行下一步")
                time.sleep(final_wait)
                run_next_step()
                break
        else:
            print(f"[{datetime.now()}] 当前空闲内存不足 {threshold_mb} MB，继续等待")
            consecutive_count = 0
            time.sleep(initial_interval)



# wait_for_memory_with_frequent_checks(device_id=1)

def plot_data_distribution(train_labels, test_labels, label_names=None):
    """
    显示训练集和测试集的标签分布情况。

    Args:
        train_labels (np.ndarray): 训练集的标签，形状为 (num_samples, num_labels)。
        test_labels (np.ndarray): 测试集的标签，形状为 (num_samples, num_labels)。
        label_names (list): 标签名称列表，默认为 None。
    """
    # 统计正样本的数量
    train_positive_counts = train_labels.sum(axis=0)
    test_positive_counts = test_labels.sum(axis=0)

    # 计算每个标签的样本总数
    num_train_samples = train_labels.shape[0]
    num_test_samples = test_labels.shape[0]

    # 计算负样本的数量
    train_negative_counts = num_train_samples - train_positive_counts
    test_negative_counts = num_test_samples - test_positive_counts

    # 设置标签名称
    num_labels = train_labels.shape[1]
    if label_names is None:
        label_names = [f"Label {i}" for i in range(num_labels)]

    # 绘制分布图
    x = np.arange(num_labels)
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    # 训练集分布
    ax.bar(x - bar_width / 2, train_positive_counts, bar_width, label='Train Positive', color='blue')
    ax.bar(x - bar_width / 2, train_negative_counts, bar_width, bottom=train_positive_counts, label='Train Negative', color='lightblue')

    # 测试集分布
    ax.bar(x + bar_width / 2, test_positive_counts, bar_width, label='Test Positive', color='orange')
    ax.bar(x + bar_width / 2, test_negative_counts, bar_width, bottom=test_positive_counts, label='Test Negative', color='peachpuff')

    # 设置图表信息
    ax.set_xlabel('Labels')
    ax.set_ylabel('Sample Counts')
    ax.set_title('Data Distribution of Train and Test Sets')
    ax.set_xticks(x)
    ax.set_xticklabels(label_names, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.show()

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
csv_file_pathwawy ='/mnt/data16t/yuy/pathway/label/pathway_3.csv'

dict_gene = read_csv_gene(csv_file_gene)
dict_pathway = read_csv_pathway(csv_file_pathwawy)

###################################################################      Must define task and molecular profile name here (in order)   ######################################################################
pathway_task_list = ['Gene']
pathway = ['R-HSA-8948216']#hsa04012\hsa04014\ hsa05161\hsa04310\hsa04151\hsa04820\hsa04512\R-HSA-8874081, R-HSA-8875878, R-HSA-6806834\R-HSA-1474290, R-HSA-8948216\R-HSA-1442490\R-HSA-1474228\hsa04020\R-HSA-1474244\R-HSA-1474290\R-HSA-8948216\hsa04814\R-HSA-3000171\R-HSA-3906995
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

batch_size = 2  # 64
node_num = 1000
init_lr = 1e-3  # 30
wd = 1e-5  # 1e-2 # weight decay
fold_num = 5
num_epochs = 128


###################################################################      Train and Test Function   ######################################################################
def add_gaussian_noise(data, noise_std=0.1):
    """
    为节点特征添加高斯噪声
    """
    noise = torch.randn_like(data.x) * noise_std
    data.x += noise
    return data

def random_node_feature_masking(data, masking_prob=0.2):
    """
    随机掩码节点特征，类似于 Dropout
    """
    mask = (torch.rand(data.x.size(0)) > masking_prob).float().view(-1, 1)
    data.x *= mask
    return data

def random_edge_dropout(data, dropout_prob=0.2):
    """
    随机丢弃图的边
    """
    edge_mask = (torch.rand(data.edge_index.size(1)) > dropout_prob).float()
    data.edge_index = data.edge_index[:, edge_mask.bool()]
    return data

def augment_dataset(dataset, noise_std=0.1, masking_prob=0.2, dropout_prob=0.2):
    """
    对数据集进行增强，应用高斯噪声、节点特征掩码和边丢弃
    """
    augmented_dataset = []
    for data in dataset:
        data = add_gaussian_noise(data, noise_std=noise_std)
        data = random_node_feature_masking(data, masking_prob=masking_prob)
        data = random_edge_dropout(data, dropout_prob=dropout_prob)
        augmented_dataset.append(data)
    return augmented_dataset

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


def pad_last_batch(data_list, batch_size):
    num_data = len(data_list)
    # 计算需要填充的数量
    padding_needed = (batch_size - (num_data % batch_size)) % batch_size

    # 创建填充的 Data 对象，确保属性与原始数据一致
    padded_data_list = data_list + [
        Data(x=torch.zeros(1, data_list[0].x.size(1)),
             edge_index=torch.empty((2, 0), dtype=torch.long),
             data_name=torch.tensor([-1]),
             y=torch.tensor([-1])  # 用 -1 或其他值表示填充数据
             ) for _ in range(padding_needed)
    ]

    return padded_data_list

def remove_padding(data_list):
    cleaned_data = []
    removed_indices = []

    for idx, data in enumerate(data_list):
        # 确保 y 是单个标量，或直接比较 tensor
        if data.y.shape == torch.Size([1]) and data.y.item() != -1:
            cleaned_data.append(data)
        elif data.y.shape != torch.Size([1]):  # 处理多元素张量的情况
            # 这里可以根据具体需求调整比较方式
            if data.y[0].item() != -1:  # 假设比较第一个元素
                cleaned_data.append(data)
        else:
            removed_indices.append(idx)  # 记录移除的索引

    return cleaned_data, removed_indices


def train(train_loader, epoch, model, optimizer, device, init_lr, num_epochs, index_hub, lambda_reg=0.03):
    model.train()
    loss_all = 0
    correct_samples = 0
    total_samples = 0
    total_labels = 0
    macro_correct = torch.zeros(len(train_loader.dataset[0].y))  # Assuming all samples have same number of labels
    macro_correct = macro_correct.to(device)
    hamming_loss_sum = 0
    correct = 0
    all_labels = []  # Collect all true labels for AUC calculation
    all_probs = []   # Collect all probabilities for AUC calculation
    save_Gradient = []
    epoch_score = []
    epoch_y = []
    epoch_name = []

    accumulation_steps = 4  # 设置梯度累积的步数
    optimizer.zero_grad()   # 初始化梯度

    for i, data_list in enumerate(train_loader):
        data_list = pad_last_batch(data_list, batch_size)

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

        output, _ , _, self_graph = model(data_list)
        data_list, removed_indices = remove_padding(data_list)
        mask = torch.ones(output.size(0), dtype=torch.bool)
        mask[removed_indices] = False
        result_output = output[mask]

        y = torch.cat([data.y.unsqueeze(0) for data in data_list]).to(output.device)
        y = y.view(result_output.size()).float()

        # 计算误差
        error = torch.abs(result_output - y)
        adjustment = error * self_graph.edge_attr.T
        adjusted_error = adjustment.sum(dim=0)

        # 确保最大值在1的范围内进行等比例缩放
        max_abs_error = adjusted_error.abs().max()
        if max_abs_error > 0:
            adjusted_error = adjusted_error / max_abs_error

        # 计算adjusted_error的长度
        length = adjusted_error.numel()
        adjust_factors = torch.full((num_classes, 1), 1 / length, dtype=torch.float32)

        new_value = 2 / length
        new_values = torch.full((len(index_hub), 1), new_value, dtype=torch.float32)
        for index, value in zip(index_hub, new_values):
            adjust_factors[index] = value
        adjust_factors = adjust_factors.to(device)
        adjusted_error = adjusted_error.to(device)

        updated_edge_attr = self_graph.edge_attr.clone()
        for i in range(len(adjust_factors)):
            updated_edge_attr[i] += adjust_factors[i] * adjusted_error[i]

        # 更新 self_graph.edge_attr
        self_graph.edge_attr = updated_edge_attr

        # 计算正则化项（L2 范数）
        l2_reg = lambda_reg * torch.norm(self_graph.edge_attr, p=2)

        result_output = result_output.to(device)
        y = y.to(device)

        # 计算损失
        loss = loss_fn(result_output, y)
        loss += l2_reg  # 添加正则化项
        loss = loss / accumulation_steps  # 归一化损失
        loss.backward()  # 反向传播，累积梯度

        # 每 accumulation_steps 步更新一次模型参数
        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()  # 更新模型参数
            optimizer.zero_grad()  # 清空梯度

        # 记录损失和其他指标
        loss_all += loss.item() * accumulation_steps  # 恢复损失值

        probs = torch.sigmoid(result_output)
        thresholds = torch.full(probs.shape, 0.5, device=probs.device)
        pred = (probs > thresholds).float()  # 使用阈值0.5二值化

        # 样本级别准确率
        correct_per_sample = (pred == y).all(dim=1).float()
        correct_samples += correct_per_sample.sum().item()
        total_samples += len(correct_per_sample)

        # 宏平均准确率
        correct_per_label = (pred == y).float().mean(dim=0)
        macro_correct += correct_per_label

        # 微平均准确率和Hamming Loss
        total_labels += y.numel()
        hamming_loss_sum += (pred != y).float().sum().item()

        # Collect all labels and predictions for AUC calculation
        all_labels.append(y.cpu())
        all_probs.append(probs.cpu())

        epoch_score.extend(result_output)
        epoch_y.extend(y)
        epoch_name.extend(detail)

    # 如果最后剩余的步数不足 accumulation_steps，手动更新一次
    if len(train_loader) % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    # Concatenate all true labels and probabilities across batches
    all_labels = torch.cat(all_labels, dim=0).detach().numpy()
    all_probs = torch.cat(all_probs, dim=0).detach().numpy()

    # Calculate per-label accuracy
    label_acc = (all_labels == (all_probs > 0.5)).mean(axis=0)

    # Calculate per-label AUC (handle cases where AUC cannot be calculated)
    label_auc = []
    for i in range(all_labels.shape[1]):
        try:
            auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
        except ValueError:
            auc = None  # AUC cannot be computed if only one class is present
        label_auc.append(auc)

    # 计算整体的指标
    accuracy_sample = correct_samples / total_samples  # 样本级别准确率
    accuracy_macro = macro_correct.mean().item() / len(train_loader)  # 宏平均准确率
    accuracy_micro = (total_labels - hamming_loss_sum) / total_labels  # 微平均准确率
    hamming_loss = hamming_loss_sum / total_labels  # Hamming Loss

    epoch_data = {
        'score': epoch_score,
        'y': epoch_y,
        'name': epoch_name
    }

    return loss_all / len(train_loader), accuracy_sample, accuracy_macro, accuracy_micro, hamming_loss, label_acc, label_auc, save_Gradient, epoch_data


def test(loader, slide_name_order, dic_slide_patch, dataset_name,lambda_reg=0.03):
    model.eval()
    loss_all = 0
    correct = 0
    epoch_pred = []
    epoch_label = []
    epoch_score = []
    dic_slide_select_node = {}
    epoch_score_save = []
    epoch_y = []
    epoch_name = []

    correct_sample = 0
    total_sample = 0
    accuracy_sample_sig_list = []

    with torch.no_grad():  # 测试时不需要计算梯度
        for data_list in loader:

            # detail = [data.data_name for data in data_list]

            detail = []
            data_list = pad_last_batch(data_list, batch_size)
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

            # 前向传播并获取预测结果和选定节点索引
            output, select_index_list, added_node_features, self_graph = model(data_list)
            data_list, removed_indices = remove_padding(data_list)
            mask = torch.ones(output.size(0), dtype=torch.bool)
            mask[removed_indices] = False
            result_output = output[mask]
            y = torch.cat([data.y.unsqueeze(0) for data in data_list]).to(output.device)
            y = y.view(result_output.size()).float()

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

            epoch_score_save.extend(result_output)
            epoch_y.extend(y)
            epoch_name.extend(detail)

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

    epoch_data = {
        'score': epoch_score_save,
        'y': epoch_y,
        'name': epoch_name
    }

    # 返回结果，包括各种评估指标
    return loss_all / len(loader),sample_acc, micro_acc, macro_acc, hamming, epoch_pred, epoch_score, epoch_label, dic_slide_select_node, label_acc, label_auc, epoch_data



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

# slide_name_order_list = [slide_name_order1]
# dic_slide_patch_list = [dic_slide_patch1]
slide_name_order_list = [slide_name_order1, slide_name_order2, slide_name_order3, slide_name_order4, slide_name_order5]
dic_slide_patch_list = [dic_slide_patch1, dic_slide_patch2, dic_slide_patch3, dic_slide_patch4, dic_slide_patch5]
# slide_name_order_list = [slide_name_order1, slide_name_order2]
# dic_slide_patch_list = [dic_slide_patch1, dic_slide_patch2]

slide_name_order_list = [list(chain.from_iterable(slide_name_order_list))]
# dic_slide_patch_list = [{**dic_slide_patch1, **dic_slide_patch2}]
dic_slide_patch_list = [{**dic_slide_patch1, **dic_slide_patch2, **dic_slide_patch3, **dic_slide_patch4, **dic_slide_patch5}]
###################################################################      Training start here    ######################################################################

result_gene = {}  # {'gene' : {1:{1:[acc,auc,....], 2:[...]}, 2:{...}}}
gene_ensemble_result = {}
# start = time.time()

save_train_num = []
# task

# for gene, task in zip(pathway, pathway_task_list):
#     if task == 'MSI':
#         label_path = 'MSI_MSS_coad.json'
#         with open(label_path) as f:
#             dic_id_label = json.load(f)
#
#     else:
#         if task == 'Protein':
#             label_path = '/mnt/data16t/yuy/data/top20_LUAD_protein.csv'
#         elif task == 'CNA':
#             label_path = '/mnt/data16t/yuy/data/top20_LUAD_CNA.csv'
#         else:
#             label_path = '/mnt/data16t/yuy/data/top20_LUAD.csv'
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


    def check_multilabel_balance(train_dataset):
        """
        检查多标签分类数据集中标签的平衡性。

        参数:
        train_dataset (list of torch_geometric.data.Data): 图数据集，每个元素是一个 Data 对象。

        返回:
        dict: 包含标签出现次数的字典和平衡性评估。
        """
        # 初始化标签计数器
        label_counts = defaultdict(int)
        total_samples = len(train_dataset)

        # 统计每个标签出现的次数
        for data in train_dataset:
            labels = data.y.numpy()  # 将标签转换为 NumPy 数组
            # 获取标签中为1的位置
            positive_labels = np.where(labels == 1)[0]
            for label in positive_labels:
                label_counts[label] += 1

        # 计算每个标签的频率
        label_frequencies = {}
        for label, count in label_counts.items():
            label_frequencies[label] = count / total_samples

        # 评估数据集的平衡性
        min_frequency = min(label_frequencies.values())
        max_frequency = max(label_frequencies.values())
        balance_score = (max_frequency - min_frequency) / max_frequency

        # 输出平衡性评估
        print("数据集标签平衡性评估：")
        print(f"最小频率标签：{min(label_frequencies, key=label_frequencies.get)} ({min_frequency:.2%})")
        print(f"最大频率标签：{max(label_frequencies, key=label_frequencies.get)} ({max_frequency:.2%})")
        print(f"平衡性得分：{(1 - balance_score):.2%} (接近1表示更加平衡)")

        # 排序
        sorted_label_frequencies = sorted(label_frequencies.items(), key=lambda item: item[1])

        # 打印排序后的标签频率
        print("标签按频率排序：")
        for label, freq in sorted_label_frequencies:
            print(f"标签 {label}: {freq:.2%}")

        # 绘制条形图
        plt.figure(figsize=(10, 6))
        plt.bar(*zip(*sorted_label_frequencies), tick_label=[f"Label {l}" for l, _ in sorted_label_frequencies])
        plt.xlabel('Labels')
        plt.ylabel('Frequency')
        plt.title('Label Frequencies')
        plt.show()

        return label_frequencies, balance_score


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


    def init_weights(m):
        print(f"Initializing {type(m)}...")
        try:
            if isinstance(m, nn.Linear):
                if hasattr(m, 'weight') and m.weight is not None:
                    init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.zeros_(m.bias)

            elif isinstance(m, GCNConv):
                if hasattr(m, 'lin') and hasattr(m.lin, 'weight') and m.lin.weight is not None:
                    init.xavier_uniform_(m.lin.weight)
                if hasattr(m.lin, 'bias') and m.lin.bias is not None:
                    init.zeros_(m.lin.bias)

            elif isinstance(m, GATConv):
                # 初始化 lin_src 和 lin_dst
                if hasattr(m, 'lin_src') and hasattr(m.lin_src, 'weight') and m.lin_src.weight is not None:
                    init.xavier_uniform_(m.lin_src.weight)
                if hasattr(m.lin_src, 'bias') and m.lin_src.bias is not None:
                    init.zeros_(m.lin_src.bias)

                if hasattr(m, 'lin_dst') and hasattr(m.lin_dst, 'weight') and m.lin_dst.weight is not None:
                    init.xavier_uniform_(m.lin_dst.weight)
                if hasattr(m.lin_dst, 'bias') and m.lin_dst.bias is not None:
                    init.zeros_(m.lin_dst.bias)

                # 初始化注意力参数
                if hasattr(m, 'att_src') and m.att_src is not None:
                    init.xavier_uniform_(m.att_src)
                if hasattr(m, 'att_dst') and m.att_dst is not None:
                    init.xavier_uniform_(m.att_dst)
                if hasattr(m, 'att_edge') and m.att_edge is not None:
                    init.xavier_uniform_(m.att_edge)

                # 初始化偏置
                if hasattr(m, 'bias') and m.bias is not None:
                    init.zeros_(m.bias)

            elif isinstance(m, (nn.Conv2d, nn.Conv1d)):
                if hasattr(m, 'weight') and m.weight is not None:
                    init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    init.zeros_(m.bias)

            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                if hasattr(m, 'weight') and m.weight is not None:
                    init.ones_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.zeros_(m.bias)

        except Exception as e:
            print(f"Error initializing {type(m)}: {e}")
            # 可以选择不 raise，继续初始化其他模块
            # raise


    ##############################################################################################
    name_negs1, name_poss1, dic_name_datas1 = add_labels(dic_name_data1, dic_id_label_list)
    name_negs2, name_poss2, dic_name_datas2 = add_labels(dic_name_data2, dic_id_label_list)
    name_negs3, name_poss3, dic_name_datas3 = add_labels(dic_name_data3, dic_id_label_list)
    name_negs4, name_poss4, dic_name_datas4 = add_labels(dic_name_data4, dic_id_label_list)
    name_negs5, name_poss5, dic_name_datas5 = add_labels(dic_name_data5, dic_id_label_list)
    # name_neg2, name_pos2, dic_name_data2 = add_label(dic_name_data2, dic_id_label)
    # name_neg3, name_pos3, dic_name_data3 = add_label(dic_name_data3, dic_id_label)
    # name_neg4, name_pos4, dic_name_data4 = add_label(dic_name_data4, dic_id_label)
    # name_neg5, name_pos5, dic_name_data5 = add_label(dic_name_data5, dic_id_label)

    s_name_pos = []
    # s_name_pos.extend(name_poss1)
    # s_name_pos.extend(name_poss2)
    # s_name_pos.extend(name_poss3)
    # s_name_pos.extend(name_poss4)
    # s_name_pos.extend(name_poss5)
    s_name_pos = [[*a, *b, *c, *d, *e] for a, b, c, d, e in
                zip(name_poss1, name_poss2, name_poss3, name_poss4, name_poss5)]
    # s_name_pos = [[item for sublist in s_name_pos for item in sublist]]
    s_name_poss = [name_poss1, name_poss2, name_poss3, name_poss4, name_poss5]
    # s_name_pos = [[*a, *b] for a, b in
    #               zip(name_poss1, name_poss2)]
    # s_name_poss = [name_poss1, name_poss2]
    s_name_poss = [[item for sublist in s_name_poss for item in sublist]]

    # name_pos.extend(name_pos2)
    # name_pos.extend(name_pos3)
    # name_pos.extend(name_pos4)
    # name_pos.extend(name_pos5)
    # name_poss = [name_pos1, name_pos2, name_pos3, name_pos4, name_pos5]
    # s_name_poss = [name_poss1]

    s_name_neg = []
    # s_name_neg.extend(name_negs1)
    # s_name_neg.extend(name_negs2)
    # s_name_neg.extend(name_negs3)
    # s_name_neg.extend(name_negs4)
    # s_name_neg.extend(name_negs5)
    s_name_neg = [[*a, *b, *c, *d, *e] for a, b, c, d, e in
                zip(name_negs1, name_negs2, name_negs3, name_negs4, name_negs5)]
    s_name_negs = [name_negs1, name_negs2, name_negs3, name_negs4, name_negs5]
    # s_name_neg = [[*a, *b] for a, b in
    #               zip(name_negs1, name_negs2)]
    # s_name_negs = [name_negs1, name_negs2]
    # s_name_negs = [chain.from_iterable(name_negs1), chain.from_iterable(name_negs2), chain.from_iterable(name_negs3),chain.from_iterable(name_negs4), chain.from_iterable(name_negs5)]
    s_name_negs = [[item for sublist in s_name_negs for item in sublist]]
    # name_neg.extend(name_neg2)
    # name_neg.extend(name_neg3)
    # name_neg.extend(name_neg4)
    # name_neg.extend(name_neg5)
    # name_negs = [name_neg1, name_neg2, name_neg3, name_neg4, name_neg5]
    # s_name_negs1 = [name_negs1]

    # dic_name_data_list = [dic_name_data1]

    # dic_name_data_list = [dic_name_datas1, dic_name_datas2]
    # dic_name_data_list = {**dic_name_datas1, **dic_name_datas2}
    dic_name_data_list = [dic_name_data1, dic_name_data2, dic_name_data3, dic_name_data4, dic_name_data5]
    dic_name_data_list ={**dic_name_datas1, **dic_name_datas2, **dic_name_datas3, **dic_name_datas4, **dic_name_datas5}

    dic_multi_model_pred = {}

    k = 1
    result_k = {}

    m = 1
    result_m = {}

    for slide_name_order1, dic_slide_patch1 in zip(slide_name_order_list, dic_slide_patch_list):
        slide_name_order = slide_name_order1
        dic_slide_patch = dic_slide_patch1

    # 提取所有样本ID和对应的标签
    samples = list(dic_name_data_list.keys())
    labels = []

    # 过滤掉没有有效 y 属性的样本
    useful_samples = []  # 存储有效的样本ID
    for sample_id in samples:
        data = dic_name_data_list[sample_id]
        if hasattr(data, 'y') and data.y is not None:
            labels.append(data.y.numpy())  # 只有当 y 存在时才调用 .numpy()
            useful_samples.append(sample_id)  # 记录有效的样本ID
        else:
            print(f"Warning: No valid 'y' attribute for sample {sample_id}. This sample will be discarded.")

    labels = np.array(labels)

    # 设置交叉验证参数
    n_splits = 5  # 例如五折交叉验证
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # 设置最小样本数量 (例如取正样本中位数)
    desired_min_count = np.median(
        [len([j for j in range(len(useful_samples)) if labels[j, i] == 1]) for i in range(labels.shape[1])])

    num_classes = len(gene_list)
    edge_values = torch.full((num_classes, 1), 1 / num_classes, dtype=torch.float32)
    new_value = 2 / num_classes  # 你想要的新值
    new_values = torch.full((len(index_hub), 1), new_value, dtype=torch.float32)
    for index, value in zip(index_hub, new_values):
        edge_values[index] = value
    model = Net(num_classes, batch_size, edge_values)  # .to(device)
    model.apply(init_weights)
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = DataParallel(model, device_ids=[2])
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=wd)

    # 统计每个标签的正样本和负样本数量
    pos_counts = labels.sum(axis=0)  # 每个标签的正样本总数
    neg_counts = labels.shape[0] - pos_counts  # 每个标签的负样本总数

    neg_counts_tensor = torch.tensor(neg_counts, dtype=torch.float32).to(device)
    pos_counts_tensor = torch.tensor(pos_counts, dtype=torch.float32).to(device)

    # 计算 pos_weight
    pos_weight = torch.log(neg_counts_tensor / (pos_counts_tensor + 1e-8) + 1)

    # 提升权重的因子
    boost_factor = 2.5

    # 修改对应标签的权重
    pos_weight[index_hub] *= boost_factor

    # 定义 BCEWithLogitsLoss，传入固定的 pos_weight
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 加载检查点
    start_fold, start_epoch = load_checkpoint(model, optimizer)

    # 用于保存每个 fold 和 epoch 的数据
    all_fold_epoch_data = {}

    # 用于保存训练日志
    save_train_num = []

    # 如果有检查点，加载之前保存的日志
    log_file_path = '/mnt/data16t/yuy/pathway/patch/only_change/GAT/train_output_RHSA8948216.txt'
    if os.path.exists(log_file_path) and start_fold > 0:
        with open(log_file_path, 'r') as f:
            save_train_num = f.read().splitlines()

    # 尝试加载 fold 索引
    fold_indices = load_fold_indices(useful_samples)
    if fold_indices is None:
        fold_indices = list(kf.split(useful_samples, labels.argmax(axis=1)))
        save_fold_indices(fold_indices, useful_samples)

    # 加载检查点
    start_fold, start_epoch = load_checkpoint(model, optimizer)

    # 设置从哪个 fold 开始训练（默认为 0，即从头开始）
    start_from_fold = 4  # 可以通过命令行参数或配置文件设置

    # 如果指定了从某个 fold 开始训练，则覆盖 start_fold
    if start_from_fold > start_fold:
        start_fold = start_from_fold

    # 遍历每个 fold
    for fold, (train_idx, val_idx) in enumerate(fold_indices):
        if fold < start_fold:
            print(f"Skipping Fold {fold + 1} as requested.")
            continue

        print(f"\nFold {fold + 1}:")

        # 获取训练集和验证集的样本ID
        train_samples = [useful_samples[i] for i in train_idx]
        val_samples = [useful_samples[i] for i in val_idx]

        # 提取train_data和val_data的标签
        train_labels = np.array([dic_name_data_list[sample_id].y.numpy() for sample_id in train_samples])
        val_labels = np.array([dic_name_data_list[sample_id].y.numpy() for sample_id in val_samples])

        # 逐标签平衡正负样本并确保每类样本差异尽量小
        for i in range(train_labels.shape[1]):  # 针对每个标签
            positive_samples = [train_samples[j] for j in range(len(train_samples)) if train_labels[j, i] == 1]
            negative_samples = [train_samples[j] for j in range(len(train_samples)) if train_labels[j, i] == 0]

            positive_count = len(positive_samples)
            negative_count = len(negative_samples)

            # 找到最大样本数量作为阈值
            max_count = max(positive_count, negative_count)

            # 如果正样本数量非常少，可以进一步增加到一个设定的最小数量（中位数）
            if positive_count < desired_min_count:
                deficit = desired_min_count - positive_count
                additional_samples_idx = np.random.choice(range(len(positive_samples)), size=int(deficit), replace=True)
                additional_samples = [positive_samples[idx] for idx in additional_samples_idx]
                train_samples.extend(additional_samples)
                # 同时扩增train_labels
                additional_labels = [train_labels[train_samples.index(sample)] for sample in additional_samples]
                train_labels = np.vstack([train_labels, np.array(additional_labels)])
                # print(f"Balanced label {i}: Added {deficit} positive samples to reach desired_min_count.")

            # 如果正样本少于负样本，进行过采样
            if positive_count < negative_count:
                deficit = negative_count - positive_count
                max_deficit = int(positive_count * 4)  # 限制正样本数量为原正样本的 1.5 倍
                deficit = min(deficit, max_deficit - positive_count)  # 控制过采样数量
                additional_samples_idx = np.random.choice(range(len(positive_samples)), size=deficit, replace=True)
                additional_samples = [positive_samples[idx] for idx in additional_samples_idx]
                train_samples.extend(additional_samples)

                # 同时扩增 train_labels
                additional_labels = [train_labels[train_samples.index(sample)] for sample in additional_samples]
                train_labels = np.vstack([train_labels, np.array(additional_labels)])

            # 如果负样本少于正样本，进行过采样
            elif negative_count < positive_count:
                deficit = positive_count - negative_count
                max_deficit = int(negative_count * 4)  # 限制负样本数量为原负样本的 1.5 倍
                deficit = min(deficit, max_deficit - negative_count)  # 控制过采样数量
                additional_samples_idx = np.random.choice(range(len(negative_samples)), size=deficit, replace=True)
                additional_samples = [negative_samples[idx] for idx in additional_samples_idx]
                train_samples.extend(additional_samples)

                # 扩增 train_labels
                additional_labels = [train_labels[train_samples.index(sample)] for sample in additional_samples]
                train_labels = np.vstack([train_labels, np.array(additional_labels)])

            # # 如果正样本少于负样本，进行过采样
            # if positive_count < negative_count:
            #     deficit = negative_count - positive_count  # 计算需要增加的正样本数量
            #     additional_samples_idx = np.random.choice(range(len(positive_samples)), size=int(deficit), replace=True)
            #     additional_samples = [positive_samples[idx] for idx in additional_samples_idx]
            #     train_samples.extend(additional_samples)
            #     # 同时扩增train_labels
            #     additional_labels = [train_labels[train_samples.index(sample)] for sample in additional_samples]
            #     train_labels = np.vstack([train_labels, np.array(additional_labels)])
            #     print(f"Balanced label {i}: Added {deficit} positive samples to match negative samples.")
            #     # 如果负样本少于正样本，进行过采样
            # elif negative_count < positive_count:
            #     deficit = positive_count - negative_count  # 计算需要增加的负样本数量
            #     additional_samples_idx = np.random.choice(range(len(negative_samples)), size=int(deficit), replace=True)
            #     additional_samples = [negative_samples[idx] for idx in additional_samples_idx]
            #     train_samples.extend(additional_samples)
            #
            #     # 扩增train_labels
            #     additional_labels = [train_labels[train_samples.index(sample)] for sample in additional_samples]
            #     train_labels = np.vstack([train_labels, np.array(additional_labels)])
            #     print(f"Balanced label {i}: Added {deficit} negative samples to match positive samples.")

            # # 限制每个标签类别的样本数量不超过最大样本数量
            # current_positive_count = len(
            #     [sample for sample in train_samples if train_labels[train_samples.index(sample), i] == 1])
            # current_negative_count = len(
            #     [sample for sample in train_samples if train_labels[train_samples.index(sample), i] == 0])
            #
            # # 如果正样本过多，裁剪正样本数量
            # if current_positive_count > max_count:
            #     train_samples = [sample for sample in train_samples if
            #                      train_labels[train_samples.index(sample), i] == 1][:max_count] + \
            #                     [sample for sample in train_samples if
            #                      train_labels[train_samples.index(sample), i] == 0]
            #     # 同时裁剪train_labels
            #     train_labels = train_labels[:len(
            #         [sample for sample in train_samples if train_labels[train_samples.index(sample), i] == 1]) + len(
            #         [sample for sample in train_samples if train_labels[train_samples.index(sample), i] == 0])]
            #     print(f"Balanced label {i}: Limited positive samples to max count {max_count}.")
            #
            # # 如果负样本过多，裁剪负样本数量
            # if current_negative_count > max_count:
            #     train_samples = [sample for sample in train_samples if
            #                      train_labels[train_samples.index(sample), i] == 1] + \
            #                     [sample for sample in train_samples if
            #                      train_labels[train_samples.index(sample), i] == 0][:max_count]
            #     # 同时裁剪train_labels
            #     train_labels = train_labels[:len(
            #         [sample for sample in train_samples if train_labels[train_samples.index(sample), i] == 1]) + len(
            #         [sample for sample in train_samples if train_labels[train_samples.index(sample), i] == 0])]
            #     print(f"Balanced label {i}: Limited negative samples to max count {max_count}.")

        # 计算训练集中的样本权重
        class_counts = np.bincount(train_labels.argmax(axis=1))  # 计算每个类的样本数
        class_weights = 1.0 / class_counts  # 计算每个类的权重（反向比例）
        sample_weights = class_weights[train_labels.argmax(axis=1)]  # 为每个样本分配权重

        # 使用 WeightedRandomSampler 进行平衡采样
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        # 创建训练集和验证集的DataLoader
        train_dataset = [dic_name_data_list[sample_id] for sample_id in train_samples]
        test_dataset = [dic_name_data_list[sample_id] for sample_id in val_samples]

        augmented_train_dataset = augment_dataset(train_dataset, noise_std=0.15, masking_prob=0.2, dropout_prob=0.2)
        augmented_test_dataset = augment_dataset(test_dataset, noise_std=0.15, masking_prob=0.2, dropout_prob=0.2)

        plot_data_distribution(train_labels, val_labels, gene_list)

        # train_dataset = pad_last_batch(train_dataset, batch_size)
        # test_dataset = pad_last_batch(test_dataset, batch_size)
        # test_loader = graphDataLoader(test_dataset, batch_size = 64, shuffle = True)#, num_workers=32)
        # train_loader = DataListLoader(train_dataset, batch_size=batch_size, sampler=sampler)  # , num_worker=32)
        train_loader = DataListLoader(augmented_train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataListLoader(test_dataset, batch_size=batch_size, shuffle=False)
        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=init_lr * 3,
                                                        steps_per_epoch=len(train_loader),
                                                        epochs=num_epochs)


        best_test_acc = test_acc = best_test_auc = best_auc = 0
        best_test_epoch = 0
        save_pos_acc = 0
        save_neg_acc = 0
        best_auc_epoch = 0
        save_pos_acc_inauc = 0
        save_neg_acc_inauc = 0

        train_accuracy = []
        train_losses = []

        test_accuracy = []
        test_losses = []
        test_auc = []

        for epoch in range(num_epochs):
            train_loss, train_acc, train_macro_acc, train_micro_acc, train_hamming_loss, train_label_acc, train_label_auc, save_Gradient_epoch, train_epoch_data = train(train_loader, epoch, model, optimizer, device, init_lr, num_epochs, index_hub)
            test_loss, test_acc, test_micro_acc, test_macro_acc, test_hamming_loss, pred_label, pred_score, epoch_label, epoch_slide_select_patch, test_label_acc, test_label_auc, test_epoch_data = test(
                test_loader, slide_name_order, dic_slide_patch, dataset_name='testing')

            train_epoch_data_save_dir = '/mnt/data16t/yuy/pathway/patch/only_change/GAT/train_epoch_data_RHSA8948216/'
            if not os.path.exists(train_epoch_data_save_dir):
                os.makedirs(train_epoch_data_save_dir)
            test_epoch_data_save_dir = '/mnt/data16t/yuy/pathway/patch/only_change/GAT/test_epoch_data_RHSA8948216/'
            if not os.path.exists(test_epoch_data_save_dir):
                os.makedirs(test_epoch_data_save_dir)

            torch.save(train_epoch_data, f'/mnt/data16t/yuy/pathway/patch/only_change/GAT/train_epoch_data_RHSA8948216/{fold + 1}_{epoch+ 1}.pth')
            torch.save(test_epoch_data, f'/mnt/data16t/yuy/pathway/patch/only_change/GAT/test_epoch_data_RHSA8948216/{fold + 1}_{epoch+ 1}.pth')


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
            roc_auc_macro = roc_auc_score(y, y_score, average='macro')
            # 计算微平均的 ROC AUC 分数
            roc_auc_micro = roc_auc_score(y, y_score, average='micro')
            # fpr, tpr, thresholds = metrics.roc_curve(y, y_score, pos_label=1)
            # test_auc = metrics.auc(fpr, tpr)
            # print(f'Epoch: {epoch + 1:03d} :')
            # print(
            #     f'Train Loss: {train_loss:.7f} | Train acc: {train_acc:.7f} | Train macro_acc: {train_macro_acc} | '
            #     f'Train micro_acc: {train_micro_acc} | Train Hamming Loss: {train_hamming_loss} |\nTest Loss: {test_loss:.7f} | Test Acc: {test_acc} | '
            #     f'Test Macro-Averaging Acc: {test_micro_acc} | Test Micro-Averaging Acc: {test_macro_acc} |\n'
            #     f'Test Macro-Averaging Auc: {roc_auc_macro} | Test Micro-Averaging Auc: {roc_auc_micro} |\n'
            #     f'Test Hamming Loss: {test_hamming_loss:.7f}\n'
            # )

            train_label_acc_list = train_label_acc.tolist()
            # print(f'Gene_List: {gene_list}')
            # print(f'train_label_acc: {train_label_acc_list}')
            # print(f'train_label_auc: {train_label_auc}')
            # print(f'test_label_acc: {test_label_acc}')
            # print(f'test_label_auc: {test_label_auc}')
            # print(f'TP: {TP}')
            # print(f'TN: {TN}')
            # print(f'FP: {FP}')
            # print(f'FN: {FN}')
            # print(f'Total TP:{total_TP}')
            # print(f'Total TN: {total_TN}')
            # print(f'Total FP: {total_FP}')
            # print(f'Total FN: {total_FN}')
            # print(f'Micro F1: {micro_f1}')
            # print(f'Macro F1: {macro_f1}')
            # print(f'F1 Scores per label: {f1_scores_list}')


            print()
            save_train_num.append(f'Fold: {fold + 1:03d} :')
            save_train_num.append('Epoch: {:03d} :'.format(epoch+ 1))
            save_train_num.append(f'Train Loss: {train_loss:.7f} | Train acc: {train_acc:.7f} | Train macro_acc: {train_macro_acc} | '
                f'Train micro_acc: {train_micro_acc} |\nTest Loss: {test_loss:.7f} | Test Acc: {test_acc} | '
                f'Test Macro-Averaging Acc: {test_micro_acc} | Test Micro-Averaging Acc: {test_macro_acc} |\n'
                f'Test Macro-Averaging Auc: {roc_auc_macro} | Test Micro-Averaging Auc: {roc_auc_micro} |\n'
                f'Test Hamming Loss: {test_hamming_loss:.7f}\n')
            save_train_num.append(f'Gene_List: {gene_list}')
            save_train_num.append(f'train_label_acc: {train_label_acc_list}')
            save_train_num.append(f'train_label_auc: {train_label_auc}')
            save_train_num.append(f'test_label_acc: {test_label_acc}')
            save_train_num.append(f'test_label_auc: {test_label_auc}')
            save_train_num.append(f'TP: {TP}')
            save_train_num.append(f'TN: {TN}')
            save_train_num.append(f'FP: {FP}')
            save_train_num.append(f'FN: {FN}')
            save_train_num.append(f'Total TP:{total_TP}')
            save_train_num.append(f'Total TN: {total_TN}')
            save_train_num.append(f'Total FP: {total_FP}')
            save_train_num.append(f'Total FN: {total_FN}')
            save_train_num.append(f'Micro F1: {micro_f1}')
            save_train_num.append(f'Macro F1: {macro_f1}')
            save_train_num.append(f'F1 Scores per label: {f1_scores_list}')

            save_train_num.append(save_Gradient_epoch)
            save_train_num.append('')

            save_path = '/mnt/data16t/yuy/pathway/patch/only_change/GAT/model_RHSA8948216/task_{}/gene_{}/model/flod_{}/'.format(
                task, gene, fold + 1)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(model.state_dict(), '{}epoch_{}.pth'.format(save_path, epoch + 1))

            # 每个 epoch 保存一次训练日志
            with open(log_file_path, 'a') as f:
                for item in save_train_num:
                    f.write(f"{item}\n")

            print(save_train_num)

            save_train_num.clear()

            # 保存当前训练状态
            save_checkpoint({
                'fold': fold,
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            })

            if roc_auc_macro > best_test_auc:
                best_test_acc = test_acc
                best_test_auc = roc_auc_macro
                prediction = pred_label
                score = pred_score
                dic_slide_select_patch = epoch_slide_select_patch

        k += 1

# 保存所有 fold 和 epoch 的数据
with open('/mnt/data16t/yuy/pathway/patch/only_change/GAT/all_fold_epoch_data_RHSA8948216.json', 'w') as f:
    json.dump(all_fold_epoch_data, f)

# 清理检查点文件
if os.path.isfile('/mnt/data16t/yuy/pathway/patch/only_change/GAT/checkpoint_RHSA8948216.pth.tar'):
    os.remove('/mnt/data16t/yuy/pathway/patch/only_change/GAT/checkpoint_RHSA8948216.pth.tar')



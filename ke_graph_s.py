########################################################################################
# 通过patch的形态学特征建立边关系
########################################################################################
import torch
import os
from torch_geometric.data import Data
from kenn import parsers
import Experiments.kegnn.parsers as gnn_parsers
import csv
import json
import pandas as pd
import numpy as np
import random
import shutil
import os
import shutil
from pathlib import Path
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, MessagePassing
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

def copy_image(src_image, dst_dir):
    """
    拷贝图片到目标文件夹
    参数：
    src_image: 源图片路径
    dst_dir: 目标文件夹路径
    """
    src_path = Path(src_image)
    dst_path = Path(dst_dir)

    # 检查源图片是否存在
    if not src_path.exists():
        print(f"错误：源图片 '{src_image}' 不存在！")
        return

    # 创建目标文件夹（如果不存在）
    dst_path.mkdir(parents=True, exist_ok=True)

    # 构建目标文件路径
    dst_file = dst_path / src_path.name

    # 拷贝图片
    shutil.copy2(src_path, dst_file)
    print(f"已拷贝：{src_path.name} -> {dst_file}")

# 指定文件夹路径
folder_path = '/mnt/data16t/yuy/pathway/patch/only_change/GIN_change/hsa04151_ke_output/result_dict/'
all_added_node_features_path = "/mnt/data16t/yuy/pathway/patch/only_change/GIN_change/hsa04151_ke_output/all_added_node_features.pth"
all_self_graph_path = "/mnt/data16t/yuy/pathway/patch/only_change/GIN_change/hsa04151_ke_output/all_self_graph.pth"
output_all_path = "/mnt/data16t/yuy/pathway/patch/only_change/GIN_change/hsa04151_ke_output/output_all.pth"
select_index_all_path = '/mnt/data16t/yuy/pathway/patch/only_change/GIN_change/hsa04151_ke_output/select_index_all.pth'
test_data_patch_lists_path = '/mnt/data16t/yuy/pathway/patch/only_change/GIN_change/hsa04151_ke_output/test_data_patch.pth'
device = torch.device('cuda:2'
                              if torch.cuda.is_available() else 'cpu')

###################################################################      Must define task and molecular profile name here (in order)   ######################################################################
pathway_task_list = ['Gene']
pathway = ['hsa04151']
csv_file_gene = '/mnt/data16t/yuy/pathway/label/gene_list.csv'
csv_file_pathwawy ='/mnt/data16t/yuy/pathway/label/pathway_new.csv'

dict_gene = read_csv_gene(csv_file_gene)
dict_pathway = read_csv_pathway(csv_file_pathwawy)
test_data_patch_lists = torch.load(test_data_patch_lists_path)
select_index_all = torch.load(select_index_all_path)
all_added_node_features = torch.load(all_added_node_features_path)
output_all = torch.load(output_all_path)
all_self_graph = torch.load(all_self_graph_path)
node_feature_list = []
patch_list = []
result_dict = {}
for node_features_adds, outputs, test_data_patches in zip(all_added_node_features, output_all, test_data_patch_lists):
    for node_feature_add, output, test_data_patch in zip(node_features_adds, outputs, test_data_patches):
        node_feature_add_expanded = node_feature_add.unsqueeze(1)
        result = torch.cat((node_feature_add_expanded, output), dim=1)  # 在列方向拼接
        node_feature_list.append(result)
        for patch in test_data_patch:
            # 提取文件名（最后一部分）
            filename = patch[0].split('/')[-1]
            # 分割键：按第一个下划线分割，取前半部分
            key = filename.split('_', 1)[0]
            # 将路径添加到字典中对应的键下
            if key not in result_dict:
                result_dict[key] = []
            result_dict[key].extend(patch)
            # patch_list.append(patch)
node_feature = torch.cat(node_feature_list)
# 获取所有.pt文件的路径
file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pt')]

# 读取每个文件并加载
all_data = {}
for file_path in file_paths:
    try:
        data = torch.load(file_path)
        # 假设data是一个字典，按key合并到all_data字典
        for key, value in data.items():
            if key not in all_data:
                all_data[key] = []  # 如果key还不存在，创建一个空列表
            all_data[key].append(value)  # 将当前file的value添加到对应的key下
    except Exception as e:
        print(f"Error loading {file_path}: {e}")

# 相似度阈值
similarity_threshold = 0.9

node_features = all_data
# 将节点特征转化为列表和张量
node_keys = list(node_features.keys())
for key,values in zip(node_keys,node_features.values()):
    for value in values:
        node_features[key] = torch.stack(value) # [num_patches, num_features]

values = list(node_features.values())
features = torch.stack(list(node_features.values()))  # [num_nodes, num_features]

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

# 1. 聚合每个节点的特征 (805, 20, 512) -> (805, 512)
# 可以使用均值聚合，也可以使用其他方法（如最大值聚合）
aggregated_features = features.mean(dim=1)  # (805, 512)

# Step 2: k近邻构建边
k = 20  # k近邻的邻居数
distances = torch.cdist(aggregated_features, aggregated_features, p=2)  # 计算欧氏距离 (805, 805)
nearest_neighbors = distances.argsort(dim=1)[:, 1:k+1]  # 每个节点找到最近的k个邻居 (805, k)

# 构建边索引 edge_index 和边权重
edge_index = []
edge_weights = []

for node, neighbors in enumerate(nearest_neighbors):
    for neighbor in neighbors:
        edge_index.append([node, neighbor.item()])  # 添加边
        edge_weights.append(distances[node, neighbor].item())  # 添加权重

edge_index = torch.tensor(edge_index, dtype=torch.long).t()  # 转置为 [2, num_edges]
edge_weights = torch.tensor(edge_weights, dtype=torch.float32)  # 转为 Tensor

# 权重归一化到 [0, 1]
edge_weights = (edge_weights - edge_weights.min()) / (edge_weights.max() - edge_weights.min())
# edge_weights = edge_weights.unsqueeze(1)  # (num_edges,) -> (num_edges, 1)
# 构建 PyG 图数据对象
data = Data(x=node_feature, edge_index=edge_index, edge_weights=edge_weights).to(device)
index1 = edge_index[0]
index2 = edge_index[1]
ke = parsers.relational_parser('logic_rules_hsa04151.txt')
node_feature = node_feature.to(device)
edge_index = edge_index.to(device)
index1 = index1.to(device)
index2 = index2.to(device)
edge_weights = edge_weights.to(device)


# ========== 全局确定性设置 ==========
def set_deterministic(seed=42):
    # PyTorch相关
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Python & NumPy
    np.random.seed(seed)
    random.seed(seed)


set_deterministic()  # 在代码开头调用


# ========== 修改后的模型类 ==========
class WeightedGCNProp(MessagePassing):
    """不可训练的带权重GCN传播层"""

    def __init__(self):
        super().__init__(aggr='add')
        # 显式初始化所有参数（即使不可训练）
        self._init_weights()

    def _init_weights(self):
        # 添加虚拟参数示例（无实际作用，仅为展示初始化方法）
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))
        torch.nn.init.constant_(self.dummy_param, 0.0)  # 固定初始化

    def forward(self, x, edge_index, edge_weights):
        row, col = edge_index
        deg = torch.bincount(row, minlength=x.size(0)).float().clamp(min=1)
        norm_weights = edge_weights / torch.sqrt(deg[row] * deg[col])
        return self.propagate(edge_index, x=x, edge_weights=norm_weights)

    def message(self, x_j, edge_weights):
        return x_j * edge_weights.view(-1, 1)


class CustomWeightedGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.gcn_prop = WeightedGCNProp()

        # 确保知识层初始化确定性
        self.ke_layer = gnn_parsers.relational_parser(knowledge_file='logic_rules_hsa04151.txt')
        self._freeze_sublayers()

    def _freeze_sublayers(self):
        # 冻结所有参数（即使某些层理论上不可训练）
        for param in self.parameters():
            param.requires_grad_(False)

    def forward(self, data):
        x, edge_index, edge_weights = data.x, data.edge_index, data.edge_weights
        x = self.gcn_prop(x, edge_index, edge_weights)
        x, _ = self.ke_layer(unary=x, edge_index=edge_index, binary=edge_weights)
        return x

model = CustomWeightedGCN()
model = model.to(device)
x_gcn_ke = model(data)

# ke = ke.to(device)
# keg = gnn_parsers.relational_parser(knowledge_file='logic_rules_hsa04151.txt')
# keg = keg.to(device)
# x_ke, _ = keg(unary=node_feature, edge_index=edge_index, binary=edge_weights)
change = []
bool_lists = []
for x_ke_single, node_feature_single, node_key in zip(x_gcn_ke, node_feature, node_keys):
    key_single = node_key.split('-')
    key_single = key_single[0:3]
    key_single = '-'.join(key_single)
    x_ke_single_pred = torch.sigmoid(x_ke_single) # 预测的概率
    node_feature_single_pred = torch.sigmoid(node_feature_single)
    x_ke_single_pred_binary = (x_ke_single_pred >= 0.5).int()
    node_feature_single_pred_binary = (node_feature_single_pred >= 0.5).int()
    change.append([x_ke_single[0], node_feature_single[0], node_key])
    try:
        dic_id_label = dic_id_label_list[key_single]
        node_feature_single_list = node_feature_single_pred_binary.tolist()
        node_feature_single_list = node_feature_single_list[1:7]
        # 将列表转换为 NumPy 数组
        dic_id_label_array = np.array(dic_id_label)
        node_feature_single_list_array = np.array(node_feature_single_list)

        # 计算准确率
        label_acc = dic_id_label_array == node_feature_single_list_array
        bool_list = label_acc.tolist()
        bool_lists.append(bool_list)
    except KeyError:
        print(f"Key '{key_single}' not found in dic_id_label_list")
        # 可以在这里处理缺失键的情况，例如赋一个默认值
        dic_id_label = 'default_label'

bool_array = np.array(bool_lists)
# 计算每个列的准确率
column_accuracy = bool_array.mean(axis=0)

sorted_by_first = sorted(change, key=lambda x: x[0].item(), reverse=True)
ke_patch_key_list = sorted_by_first[0:5]
ke_patch = []
for ke_patch_list in ke_patch_key_list:
    ke_patch_key = ke_patch_list[2]
    ke_patch.extend(result_dict[ke_patch_key])
sorted_by_second = sorted(change, key=lambda x: x[1].item(), reverse=True)
no_ke_patch_key_list = sorted_by_second[0:5]
no_ke_patch = []
for no_ke_patch_list in no_ke_patch_key_list:
    no_ke_patch_key = no_ke_patch_list[2]
    no_ke_patch.extend(result_dict[no_ke_patch_key])

for path_ke, path_no_ke in zip(ke_patch, no_ke_patch):
    src_image = path_ke
    dst_dir = '/mnt/data16t/yuy/pathway/patch/only_change/GIN_change/hsa04151_ke_output/ke_patch'
    copy_image(src_image, dst_dir)
    src_image = path_no_ke
    dst_dir = '/mnt/data16t/yuy/pathway/patch/only_change/GIN_change/hsa04151_ke_output/no_ke_patch'
    copy_image(src_image, dst_dir)

# 使用列表推导式和 zip 函数找到不一致的行
mismatched_rows = [(i, row1, row2) for i, (row1, row2) in enumerate(zip(sorted_by_first, sorted_by_second)) if row1 != row2]
print()
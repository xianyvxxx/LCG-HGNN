import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import os
from collections import defaultdict


all_added_node_features_path = "/mnt/data16t/yuy/pathway/patch/only_change/GIN_change/hsa04151_ke_output/all_added_node_features.pth"
all_self_graph_path = "/mnt/data16t/yuy/pathway/patch/only_change/GIN_change/hsa04151_ke_output/all_self_graph.pth"
output_all_path = "/mnt/data16t/yuy/pathway/patch/only_change/GIN_change/hsa04151_ke_output/output_all.pth"
select_index_all_path = "/mnt/data16t/yuy/pathway/patch/only_change/GIN_change/hsa04151_ke_output/select_index_all.pth"
test_data_patch_path = "/mnt/data16t/yuy/pathway/patch/only_change/GIN_change/hsa04151_ke_output/test_data_patch.pth"

all_added_node_features = torch.load(all_added_node_features_path)
all_self_graph = torch.load(all_self_graph_path)
output_all = torch.load(output_all_path)
select_index_all = torch.load(select_index_all_path)
test_data_patch = torch.load(test_data_patch_path)

print()


class MyDataset(Dataset):
    def __init__(self, x, transform):
        self.x = x
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.x[index]
        # print(f"Loading image from: {img_path}")
        try:
            img = Image.open(img_path)
            img = img.convert("RGB")  # 确保图像为RGB格式
        except Exception as e:
            print(f"Error loading image: {e}")
        img = img.resize((224, 224))
        if not isinstance(img, (Image.Image, np.ndarray)):
            raise TypeError(f"Expected image type, but got {type(img)}")
        img = self.transform(img)
        return img, img_path  # 返回图像和路径

    def __len__(self):
        return len(self.x)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

########################################################################################
# 抽取代表性patch的特征
########################################################################################

# use resnet18 as feature extractor
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model = torch.nn.Sequential(*list(model.children())[:-1])
model = nn.DataParallel(model, device_ids=[2])
model.to(device)

def flatten(nested_list):
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):  # 如果元素是列表，递归展开
            flat_list.extend(flatten(item))
        else:
            flat_list.append(item)  # 否则直接添加到结果列表
    return flat_list
data_list_all = flatten(test_data_patch)
dataset = MyDataset(data_list_all, transform=transform)
# 创建DataLoader
loader = DataLoader(dataset, batch_size=160, shuffle=False, num_workers=1)
all_output = []
result_dict = {}  # 存储结果的字典
counter = 0
for batch_imgs, batch_paths in loader:
    inputs = batch_imgs.to(device)  # 将数据移动到 GPU
    outputs = model(inputs).squeeze()  # 获取模型的输出

    # 处理每一个路径和输出，将其按文件名提取 key
    for path, output in zip(batch_paths, outputs):
        file_name = path.split('/')[-1]  # 提取文件名
        key = '_'.join(file_name.split('_')[:1])  # 获取类似 TCGA-44-2664-01A-01-TS1 的部分

        # 确保字典中存在该 key 并追加数据
        if key not in result_dict:
            result_dict[key] = []

        result_dict[key].append(output.cpu())  # 移动到 CPU 以便保存

    # 每个 batch 后保存字典并清空它以释放 GPU 内存
    base_dir = f'/mnt/data16t/yuy/pathway/patch/only_change/GIN_change/hsa04151_ke_output/result_dict'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    file_path = os.path.join(base_dir, f'result_dict_{counter}.pt')

    torch.save(result_dict, file_path)
    counter += 1  # 更新计数器
    # 之后再清空字典，如果需要释放内存
    result_dict.clear()

    print(counter)





print()

########################################################################################
# 节点特征填充：output_all+all_added_node_features
########################################################################################pip
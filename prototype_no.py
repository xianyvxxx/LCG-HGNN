from skimage.metrics import structural_similarity as ssim
from torchvision import models, transforms
import torch
import os
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import cupy as cp
import spams

print("PyTorch 版本:", torch.__version__)
print("CUDA 是否可用:", torch.cuda.is_available())
print("当前设备:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
print("当前设备 ID:", cp.cuda.Device().id)
print("可用设备数量:", cp.cuda.runtime.getDeviceCount())


class TissueMaskException(Exception):
    pass


def get_tissue_mask(ImgInput, luminosity_threshold=0.8):
    """生成组织掩码"""
    if isinstance(ImgInput, cp.ndarray):
        ImgInput = cp.asnumpy(ImgInput)
    Img_LAB = cv2.cvtColor(ImgInput, cv2.COLOR_RGB2LAB)
    L = Img_LAB[:, :, 0] / 255.0
    mask = L < luminosity_threshold
    if mask.sum() == 0:
        raise TissueMaskException("Empty tissue mask computed")
    return mask


def convert_RGB_to_OD(I):
    """RGB转光学密度空间"""
    I = cp.maximum(I, 1)  # 防止log(0)
    return -cp.log(I / 255.0)


def convert_OD_to_RGB(OD):
    """光学密度转RGB"""
    return (255 * cp.exp(-OD)).astype(cp.uint8)


def normalize_matrix_rows(A):
    """行归一化"""
    return A / cp.linalg.norm(A, axis=1)[:, None]


def get_stain_matrix(ImgInput, luminosity_threshold=0.8, angular_percentile=99):
    """获取染色矩阵"""
    if isinstance(ImgInput, np.ndarray):
        ImgInput = cp.array(ImgInput)

    tissue_mask = get_tissue_mask(ImgInput, luminosity_threshold)
    OD = convert_RGB_to_OD(ImgInput).reshape((-1, 3))
    OD = OD[tissue_mask.ravel()]

    # 主成分分析
    cov = cp.cov(OD, rowvar=False)
    _, V = cp.linalg.eigh(cov)
    V = V[:, [2, 1]]  # 取最后两个特征向量

    # 调整方向
    if V[0, 0] < 0: V[:, 0] *= -1
    if V[0, 1] < 0: V[:, 1] *= -1

    # 角度分布分析
    That = cp.dot(OD, V)
    phi = cp.arctan2(That[:, 1], That[:, 0])

    min_phi = cp.percentile(phi, 100 - angular_percentile)
    max_phi = cp.percentile(phi, angular_percentile)

    v1 = cp.dot(V, cp.array([cp.cos(min_phi), cp.sin(min_phi)]))
    v2 = cp.dot(V, cp.array([cp.cos(max_phi), cp.sin(max_phi)]))

    # 确定染色顺序
    HE = cp.array([v1, v2]) if v1[0] > v2[0] else cp.array([v2, v1])
    return normalize_matrix_rows(HE)


def get_concentrations(I, stain_matrix, regularizer=0.01):
    OD = convert_RGB_to_OD(I).reshape((-1, 3))
    OD = cp.asnumpy(OD)
    stain_matrix = cp.asnumpy(stain_matrix)
    result = spams.lasso(X=OD.T, D=stain_matrix.T, mode=2, lambda1=regularizer, pos=True).toarray().T
    result = cp.array(result)
    return result


def macenko_normalize(source, target, luminosity_threshold=0.8, angular_percentile=99):
    """
    Macenko颜色归一化核心函数

    Args:
        source: 参考图像 (RGB格式)
        target: 待标准化的目标图像 (RGB格式)

    Returns:
        归一化后的RGB图像 (numpy数组)
    """
    # 确保输入是cupy数组
    source_cp = cp.array(source) if isinstance(source, np.ndarray) else source
    target_cp = cp.array(target) if isinstance(target, np.ndarray) else target

    # 获取染色矩阵（使用参考图像）
    stain_matrix = get_stain_matrix(source_cp, luminosity_threshold, angular_percentile)

    # 计算目标图像的浓度
    concentrations = get_concentrations(target_cp, stain_matrix)

    # 重建图像（使用参考图像的染色矩阵）
    OD = cp.dot(concentrations, stain_matrix)
    normalized_RGB = convert_OD_to_RGB(OD.reshape(target_cp.shape))

    return cp.asnumpy(normalized_RGB)


# 基于预训练CNN的特征提取
model = models.resnet50(pretrained=True).eval()
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def extract_cnn_features(image):
    """提取CNN特征，返回一维特征向量"""
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = model.conv1(image)
        features = model.bn1(features)
        features = model.relu(features)
        features = model.maxpool(features)
        features = model.layer1(features)  # 提取浅层特征
    return features.squeeze().numpy().flatten()


def preprocess_clinical_images(clinical_dir, reference_image, cache_file=None):
    """
    预处理所有临床图像：标准化 + 特征提取

    Args:
        clinical_dir: 临床图像目录
        reference_image: 参考图像 (RGB格式)
        cache_file: 特征缓存文件路径 (可选)

    Returns:
        (clinical_features, clinical_filenames)
        clinical_features: 临床图像特征列表
        clinical_filenames: 临床图像文件名列表
    """
    # 检查缓存
    if cache_file and os.path.exists(cache_file):
        print(f"✅ 加载临床图像特征缓存: {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
        return data['features'], data['filenames']

    print("🔄 开始预处理临床图像...")
    clinical_features = []
    clinical_filenames = []

    # 处理所有临床图像
    for filename in tqdm(os.listdir(clinical_dir), desc="临床图像预处理"):
        clinical_img_path = os.path.join(clinical_dir, filename)
        clinical_img = cv2.imread(clinical_img_path)

        if clinical_img is None:
            continue

        # 转换为RGB
        clinical_img_rgb = cv2.cvtColor(clinical_img, cv2.COLOR_BGR2RGB)

        try:
            # 标准化
            normalized_image = macenko_normalize(reference_image, clinical_img_rgb)

            # 特征提取
            features = extract_cnn_features(normalized_image)
            clinical_features.append(features)
            clinical_filenames.append(filename)

        except Exception as e:
            print(f"  跳过 {filename}: {str(e)}")
            continue

    # 保存缓存
    if cache_file:
        print(f"💾 保存临床图像特征缓存到: {cache_file}")
        np.savez_compressed(cache_file,
                            features=np.array(clinical_features),
                            filenames=clinical_filenames)

    return clinical_features, clinical_filenames


def main():
    # 配置路径
    clinical_dir = "G:/data/segment/nonormlize/"
    predicted_img_PATH = "G:/data/hsa04151_ke_output/no_ke_patch/"
    output_file = r"G:\data\nonormlize_no_ke_prototype.txt"

    # 参考图像路径
    ref_path = 'E:/YUY/code/coding/coding/mycode/Review_Molecular_profile_prediction_GNN-main/1. Data_preprocessing/Ref.png'

    # 检查路径
    if not os.path.exists(clinical_dir):
        raise FileNotFoundError(f"临床图像目录不存在: {clinical_dir}")
    if not os.path.exists(predicted_img_PATH):
        raise FileNotFoundError(f"预测图像目录不存在: {predicted_img_PATH}")
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"参考图像不存在: {ref_path}")

    # 加载参考图像
    reference_image = cv2.imread(ref_path)
    if reference_image is None:
        raise ValueError(f"无法加载参考图像: {ref_path}")
    reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)

    # 预处理临床图像（只做一次）
    clinical_cache = os.path.join(os.path.dirname(output_file), "nonormlize_clinical_features.npz")
    clinical_features, clinical_filenames = preprocess_clinical_images(
        clinical_dir,
        reference_image,
        cache_file=clinical_cache
    )

    if len(clinical_features) == 0:  # 正确的检查方式
        raise ValueError("没有成功处理任何临床图像")

    print(f"\n✅ 临床图像预处理完成: {len(clinical_features)} 张图像")
    print(f"  特征维度: {clinical_features[0].shape[0]}")

    # 转换为NumPy数组以便批量计算
    clinical_features_array = np.array(clinical_features)

    # 处理所有预测图像
    results = []
    mean_sim_list = []
    for filename_PRE in tqdm(os.listdir(predicted_img_PATH), desc="处理预测图像"):
        pred_img_path = os.path.join(predicted_img_PATH, filename_PRE)
        pred_img = cv2.imread(pred_img_path)

        if pred_img is None:
            continue

        # 提取预测图像特征
        pred_features = extract_cnn_features(pred_img)

        # 计算与所有临床图像的相似度 (批量计算)
        similarities = cosine_similarity([pred_features], clinical_features_array)[0]

        # 记录结果
        results.append((filename_PRE, similarities.tolist()))

        # 打印统计信息
        mean_sim = np.mean(similarities)
        mean_sim_list.append(mean_sim)
        max_sim = np.max(similarities)
        best_match = clinical_filenames[np.argmax(similarities)]
        print(f"\n{filename_PRE} 结果:")
        print(f"  平均相似度: {mean_sim:.4f}")
        print(f"  最大相似度: {max_sim:.4f} (匹配: {best_match})")

    # 一次性写入所有结果
    with open(output_file, 'w', encoding='utf-8') as f:
        for filename_PRE, similarities in results:
            # 格式: 预测图像文件名: [sim1, sim2, ...]
            f.write(f"{filename_PRE}: {[round(s, 6) for s in similarities]}\n")

    sort_mean_sim = sorted(mean_sim_list)
    print(f"\n✅ 平均相似度排序情况:{sort_mean_sim}")

    print(f"\n✅ 所有结果已保存到: {output_file}")
    print(f"  共处理: {len(results)} 个预测图像")
    print(f"  临床图像特征缓存: {clinical_cache}")


if __name__ == "__main__":
    main()
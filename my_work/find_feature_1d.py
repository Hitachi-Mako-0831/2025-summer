import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns  # 用于更美观的统计图
import random

# 1. 加载特征向量
def load_feature_vectors(file_path):
    """从 JSON 文件加载特征向量。"""
    with open(file_path, 'r') as f:
        feature_vectors = json.load(f)
    return np.array(feature_vectors)

# 2. 加载包含 Source 信息的条目并生成标签
def load_labels_from_source(file_path):
    """从包含 Source 信息的 JSON 文件加载数据并生成标签。"""
    with open(file_path, 'r') as f:
        data = json.load(f)

    labels = []
    for item in data:
        if item.get("Source") == "human":
            labels.append(0)
        else:
            labels.append(1)
    return np.array(labels)

# 3. 数据标准化 (可选但推荐)
def standardize_data(data):
    """标准化数据（零均值和单位方差）。"""
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

# 4. 使用 PCA 降维到一维
def perform_one_dimensional_pca(data, random_state=42):
    """使用 PCA 降维到一维。"""
    pca = PCA(n_components=1, random_state=random_state)
    principal_component = pca.fit_transform(data)
    return principal_component, pca

# 5. 分析和可视化一维降维后的数据和标签
def analyze_one_dimensional_pca(principal_component, labels, save_prefix='one_dimensional_pca'):
    """分析一维 PCA 降维后的数据，并使用并排直方图根据标签进行可视化。"""
    plt.figure(figsize=(10, 6))

    human_data = principal_component[labels == 0].flatten()  # 展平为一维数组
    gpt_data = principal_component[labels == 1].flatten()    # 展平为一维数组

    plt.hist(human_data, bins=50, color='blue', alpha=0.7, label='human')
    plt.hist(gpt_data, bins=50, color='red', alpha=0.7, label='GPT')

    plt.xlabel('Principal Component 1 Value')
    plt.ylabel('Count')
    plt.title('1D PCA Projection Distribution by Source')
    plt.legend(title='Source')
    plt.grid(True, linestyle='--')

    save_path = f'{save_prefix}_histogram.png'
    plt.savefig(save_path)
    plt.show()
    print(f"1D PCA histogram plot saved to: {save_path}")

    # 打印解释的方差比例
    print(f"Explained variance ratio (1 component): {pca_model.explained_variance_ratio_[0]:.4f}")

    # (可选) 保存降维后的一维数据
    np.savetxt(f'{save_prefix}_projection.txt', principal_component)
    print(f"1D PCA projection data saved to: {save_prefix}_projection.txt")
    
    
if __name__ == "__main__":
    feature_vector_file = 'all_feature_vectors.json'  # 替换为你的特征向量文件路径
    source_info_file = 'all_rewrite_data.json'  # 替换为包含 Source 信息的 JSON 文件路径

    # 加载特征向量和标签
    feature_vectors = load_feature_vectors(feature_vector_file)
    labels = load_labels_from_source(source_info_file)

    print("Shape of feature vectors:", feature_vectors.shape)
    print("Shape of labels:", labels.shape)
    print("First few labels:", labels[:10])

    # 确保特征向量和标签的数量一致
    if feature_vectors.shape[0] != labels.shape[0]:
        raise ValueError("The number of feature vectors and labels must be the same.")

    # 标准化数据
    scaled_data, scaler = standardize_data(feature_vectors)

    # 使用 PCA 降维到一维
    principal_component, pca_model = perform_one_dimensional_pca(scaled_data)

    # 分析和可视化一维降维后的数据和标签
    analyze_one_dimensional_pca(principal_component, labels)
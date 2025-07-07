import json
import numpy as np
import matplotlib.pyplot as plt
import umap.umap_ as umap  # 导入 UMAP
from sklearn.preprocessing import StandardScaler

# 1. 加载特征向量
def load_feature_vectors(file_path):
    """从 JSON 文件加载特征向量。"""
    with open(file_path, 'r') as f:
        feature_vectors = json.load(f)
        for feature_vector in feature_vectors:
            feature_vector[:] = feature_vector[0:40] + feature_vector[-8:]
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

# 4. 使用 UMAP 进行降维
def perform_umap(data, n_components=2, random_state=42):
    """使用 UMAP 进行降维。"""
    reducer = umap.UMAP(n_components=n_components, random_state=random_state, n_neighbors = 10, init = 'spectral', metric = 'cosine', n_jobs = -1)
    embedding = reducer.fit_transform(data)
    return embedding

# 5. 可视化降维后的数据并根据标签着色
def visualize_umap_with_labels(principal_components, labels, save_path='pca_labeled.png'):
    """可视化降维后的数据，根据标签着色，并保存图像。"""
    plt.figure(figsize=(8, 6))

    # 定义颜色和标签的映射
    label_map = {0: 'human', 1: 'GPT'}
    colors = [('green' if label == 0 else 'red') for label in labels] # 自定义颜色
    unique_labels = np.unique(labels)

    # 绘制散点图
    scatter = plt.scatter(principal_components[:, 0], principal_components[:, 1], c=colors)

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('2D UMAP with Source Labels')
    plt.grid(True)

    # 创建自定义图例
    handles = [plt.Rectangle((0, 0), 1, 1, color=('green' if label == 0 else 'red')) for label in unique_labels]
    legend_labels = [label_map[label] for label in unique_labels]
    plt.legend(handles, legend_labels, title="Source")

    plt.savefig(save_path)
    plt.show()
    print(f"Labeled PCA projection image saved to: {save_path}")

if __name__ == "__main__":
    feature_vector_file = 'all_feature_vectors.json'  # 替换为你的特征向量文件路径
    source_info_file = 'all_rewrite_data.json'  # 替换为包含 Source 信息的 JSON 文件路径
    save_file = 'umap_labeled_projection.png'

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

    # 使用 UMAP 进行降维
    n_components_to_keep = 2
    umap_embedding = perform_umap(scaled_data, n_components=n_components_to_keep)

    # 可视化结果
    visualize_umap_with_labels(umap_embedding, labels, save_path=save_file)
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. 加载特征向量
def load_feature_vectors(file_path):
    """从 JSON 文件加载特征向量。"""
    with open(file_path, 'r') as f:
        feature_vectors = json.load(f)
        for feature_vector in feature_vectors:
            feature_vector = feature_vector[0:40] + feature_vector[-8:]
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

# 3. 数据标准化 (可选)
def standardize_data(data):
    """标准化数据（零均值和单位方差）。"""
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

# 4. 执行 PCA
def perform_pca(data, n_components=2):
    """执行 PCA 以提取主成分。"""
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data)
    return principal_components, pca

# 5. 可视化降维后的数据并根据标签着色
def visualize_pca_with_labels(principal_components, labels, save_path='pca_labeled.png'):
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
    plt.title('2D PCA Projection with Source Labels')
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
    label_file = 'all_rewrite_data.json'  # 替换为你的标签文件路径
    save_file = 'pca_labeled_projection.png'

    # 加载特征向量和标签
    feature_vectors = load_feature_vectors(feature_vector_file)
    labels = load_labels_from_source(label_file)

    print("Shape of feature vectors:", feature_vectors.shape)
    print("Shape of labels:", labels.shape)
    print("First few labels:", labels[:10])

    # 确保特征向量和标签的数量一致
    if feature_vectors.shape[0] != labels.shape[0]:
        raise ValueError("The number of feature vectors and labels must be the same.")

    # 标准化数据
    scaled_data, scaler = standardize_data(feature_vectors)

    # 执行 PCA
    n_components_to_keep = 2
    principal_components, pca_model = perform_pca(scaled_data, n_components=n_components_to_keep)

    # 可视化结果
    visualize_pca_with_labels(principal_components, labels, save_path=save_file)

    # 分析 PCA 结果 (可选)
    print("\nExplained variance ratio:", pca_model.explained_variance_ratio_)
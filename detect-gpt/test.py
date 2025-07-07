import random
import datasets
import os
import json # 导入 json 模块，用于读取本地json文件

# ... (保持 SEPARATOR, process_prompt, process_spaces, strip_newlines 等现有代码不变) ...

DATASETS = ['writing', 'english', 'german', 'pubmed', 'my_custom_local'] # <--- 添加你的数据集名称

# ... (保持 load_pubmed, load_writing, load_language, load_german, load_english 等现有函数不变) ...


# --- 添加一个新的函数来加载你的本地 JSON 数据集 ---
def load_my_custom_local(cache_dir=None):
    """
    加载本地的 Index, Text, Source 格式的 JSON 数据集。
    假设文件路径是 project_root/dataset/my_custom_local_data.json
    """
    # 动态构建文件路径，确保能够找到你的本地文件
    # 假设你的 my_custom_local_data.json 在项目根目录下的 dataset/ 文件夹中
    # 你可能需要根据实际的项目结构调整这个路径
    
    # 推荐：使用 os.path.join 来构建跨平台的路径
    # 这里的 __file__ 指的是 custom_datasets.py 文件本身
    # os.path.abspath(__file__) 获取 custom_datasets.py 的绝对路径
    # os.path.dirname(...) 向上移动一级，直到 project_root
    # 如果你的 dataset 文件夹和 custom_datasets.py 在同一目录下，可能不需要 os.path.dirname(os.path.dirname(...))
    
    # 请根据你实际的文件位置调整这一行
    # 例如，如果 custom_datasets.py 在 project_root/src/ 目录下，而你的数据在 project_root/dataset/
    # 那么你需要 os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset', 'my_custom_local_data.json')

    # 假设 your_main_script.py, custom_datasets.py 都在项目根目录，数据在 project_root/dataset/
    # 或者简单起见，假设你的数据文件和执行脚本在同一目录下
    # local_json_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'my_custom_local_data.json')
    
    # 针对你之前提到的 'dataset/News/news_combine.json' 的路径，你可以这样构建：
    # 这里的路径是相对于你运行 main 脚本的目录而言的
    local_json_file_path = os.path.join('dataset', 'News', 'news_combine.json')
    
    print(f"尝试从本地加载数据集: {local_json_file_path}")

    if not os.path.exists(local_json_file_path):
        raise FileNotFoundError(f"未找到本地数据集文件：{local_json_file_path}。请检查路径是否正确。")

    # 使用 datasets 库加载本地 JSON 文件
    # 它会自动解析 Index, Text, Source 字段
    loaded_data = datasets.load_dataset('json', data_files=local_json_file_path, split='train')

    # 从加载的数据集中提取我们需要的 'Text' 字段作为 generate_data 的输入
    # generate_data 期望的是一个文本列表
    # 如果你还需要其他字段（例如 Source），你需要在 generate_data 或后续函数中处理
    texts = loaded_data['Text']

    # 这里可以添加额外的处理，例如基于 Source 字段的过滤
    # For example, if you only want 'Human' generated texts initially:
    # texts = [example['Text'] for example in loaded_data if example['Source'] == 'Human']

    return texts


# ... (保持 load 函数不变) ...
def load(name, cache_dir, **kwargs):
    if name in DATASETS:
        load_fn = globals()[f'load_{name}']
        return load_fn(cache_dir=cache_dir, **kwargs)
    else:
        raise ValueError(f'Unknown dataset {name}')
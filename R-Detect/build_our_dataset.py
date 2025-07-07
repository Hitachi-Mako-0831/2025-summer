# import json
# import os
# import random

# def prepare_code_dataset(
#     gpt_json_path,
#     human_json_path,
#     output_root_dir,
#     train_ratio=0.8,
#     explanation_key="Explanation", # 包含代码解释的键
#     implementation_key="Implementation" # 包含代码实现的键
    
# ):
#     """
#     读取 code_GPT.json 和 code_human.json 文件，
#     将 Explanation 和 Implementation 字段的内容保存为 .txt 文件，
#     并按指定比例划分训练集和测试集，存储在目标目录结构中。

#     Args:
#         gpt_json_path (str): code_GPT.json 文件的路径。
#         human_json_path (str): code_human.json 文件的路径。
#         output_root_dir (str): 生成数据集的根目录。
#         train_ratio (float): 训练集与总数据的比例。
#         explanation_key (str): JSON 中包含 Explanation 文本的键名。
#         implementation_key (str): JSON 中包含 Implementation 文本的键名。
#     """
#     print(f"Starting dataset preparation for GPT: {gpt_json_path}, Human: {human_json_path}")

#     # --- 1. 读取 JSON 文件 ---
#     gpt_data = []
#     try:
#         with open(gpt_json_path, 'r', encoding='utf-8') as f:
#             gpt_data = json.load(f)
#         print(f"Loaded {len(gpt_data)} entries from {gpt_json_path}")
#     except FileNotFoundError:
#         print(f"Error: GPT JSON file not found at {gpt_json_path}")
#         return
#     except json.JSONDecodeError:
#         print(f"Error: Could not decode JSON from {gpt_json_path}")
#         return

#     human_data = []
#     try:
#         with open(human_json_path, 'r', encoding='utf-8') as f:
#             human_data = json.load(f)
#         print(f"Loaded {len(human_data)} entries from {human_json_path}")
#     except FileNotFoundError:
#         print(f"Error: Human JSON file not found at {human_json_path}")
#         return
#     except json.JSONDecodeError:
#         print(f"Error: Could not decode JSON from {human_json_path}")
#         return

#     # --- 2. 提取文本内容并合并 Explanation 和 Implementation ---
#     # 通常，对于代码检测，我们希望检测的是代码本身或者代码加注释。
#     # 这里我们选择将Explanation和Implementation合并起来作为要检测的文本。
#     # 你可以根据实际需求调整这里，例如只用Implementation。
    
#     # 获取GPT文本
#     gpt_texts = []
#     for entry in gpt_data:
#         explanation = entry.get(explanation_key, "").strip()
#         implementation = entry.get(implementation_key, "").strip()
#         combined_text = f"{explanation}\n\n{implementation}" if explanation and implementation else explanation or implementation
#         if combined_text: # 确保文本不为空
#             gpt_texts.append(combined_text)
#     print(f"Extracted {len(gpt_texts)} non-empty texts from GPT data.")

#     # 获取Human文本
#     human_texts = []
#     for entry in human_data:
#         explanation = entry.get(explanation_key, "").strip()
#         implementation = entry.get(implementation_key, "").strip()
#         combined_text = f"{explanation}\n\n{implementation}" if explanation and implementation else explanation or implementation
#         if combined_text: # 确保文本不为空
#             human_texts.append(combined_text)
#     print(f"Extracted {len(human_texts)} non-empty texts from Human data.")

#     # --- 3. 划分训练集和测试集 ---
#     random.seed(42) # 保证划分结果可复现
#     random.shuffle(gpt_texts)
#     random.shuffle(human_texts)

#     gpt_train_count = int(len(gpt_texts) * train_ratio)
#     human_train_count = int(len(human_texts) * train_ratio)

#     gpt_train_texts = gpt_texts[:gpt_train_count]
#     gpt_test_texts = gpt_texts[gpt_train_count:]

#     human_train_texts = human_texts[:human_train_count]
#     human_test_texts = human_texts[human_train_count:]

#     print(f"GPT Data: Train {len(gpt_train_texts)}, Test {len(gpt_test_texts)}")
#     print(f"Human Data: Train {len(human_train_texts)}, Test {len(human_test_texts)}")

#     # --- 4. 创建目标目录结构 ---
#     train_human_dir = os.path.join(output_root_dir, "train", "human")
#     train_machine_dir = os.path.join(output_root_dir, "train", "machine")
#     test_dir = os.path.join(output_root_dir, "test")

#     os.makedirs(train_human_dir, exist_ok=True)
#     os.makedirs(train_machine_dir, exist_ok=True)
#     os.makedirs(test_dir, exist_ok=True)
#     print(f"Created output directories under: {output_root_dir}")

#     # --- 5. 保存文本文件 ---
#     # 保存训练集
#     print("Saving training data...")
#     for i, text in enumerate(human_train_texts):
#         with open(os.path.join(train_human_dir, f"doc_human_{i}.txt"), 'w', encoding='utf-8') as f:
#             f.write(text)
#     for i, text in enumerate(gpt_train_texts):
#         with open(os.path.join(train_machine_dir, f"doc_machine_{i}.txt"), 'w', encoding='utf-8') as f:
#             f.write(text)
#     print("Training data saved.")

#     # 保存测试集
#     print("Saving testing data...")
#     for i, text in enumerate(human_test_texts):
#         with open(os.path.join(test_dir, f"doc_test_human_{i}.txt"), 'w', encoding='utf-8') as f:
#             f.write(text)
#     for i, text in enumerate(gpt_test_texts):
#         with open(os.path.join(test_dir, f"doc_test_machine_{i}.txt"), 'w', encoding='utf-8') as f:
#             f.write(text)
#     print("Testing data saved.")

#     print("\nDataset preparation complete!")
#     print(f"Generated dataset in: {os.path.abspath(output_root_dir)}")
#     print("You can now use this dataset with feature_ref_generater.py and main.py.")

# if __name__ == "__main__":
#     # 示例用法：
#     # 假设你的 code_GPT.json 和 code_human.json 在当前目录下
#     gpt_json_file = "./dataset/HumanEval Code/code_GPT.json"
#     human_json_file = "./dataset/HumanEval Code/code_human.json"
#     output_dataset_root = "./my_code_dataset" # 你希望生成的数据集根目录

#     prepare_code_dataset(
#         gpt_json_file,
#         human_json_file,
#         output_dataset_root,
#         train_ratio=0.8
#     )

import json
import os
import random

def prepare_essay_dataset(
    gpt_json_path,
    human_json_path,
    output_root_dir,
    train_ratio=0.8,
    text_key="Text" # 新增一个参数，用于指定包含文本内容的键
):
    """
    读取 essay_GPT.json 和 essay_human.json 文件，
    将 'Text' 字段的内容保存为 .txt 文件，
    并按指定比例划分训练集和测试集，存储在目标目录结构中。

    Args:
        gpt_json_path (str): essay_GPT.json 文件的路径。
        human_json_path (str): essay_human.json 文件的路径。
        output_root_dir (str): 生成数据集的根目录。
        train_ratio (float): 训练集与总数据的比例。
        text_key (str): JSON 中包含文章文本的键名。
    """
    print(f"Starting dataset preparation for GPT essay: {gpt_json_path}, Human essay: {human_json_path}")

    # --- 1. 读取 JSON 文件 ---
    gpt_data = []
    try:
        with open(gpt_json_path, 'r', encoding='utf-8') as f:
            gpt_data = json.load(f)
        print(f"Loaded {len(gpt_data)} entries from {gpt_json_path}")
    except FileNotFoundError:
        print(f"Error: GPT JSON file not found at {gpt_json_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {gpt_json_path}")
        return

    human_data = []
    try:
        with open(human_json_path, 'r', encoding='utf-8') as f:
            human_data = json.load(f)
        print(f"Loaded {len(human_data)} entries from {human_json_path}")
    except FileNotFoundError:
        print(f"Error: Human JSON file not found at {human_json_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {human_json_path}")
        return

    # --- 2. 提取文本内容 ---
    # 现在只提取 'Text' 字段
    gpt_texts = []
    for entry in gpt_data:
        text_content = entry.get(text_key, "").strip()
        if text_content: # 确保文本不为空
            gpt_texts.append(text_content)
    print(f"Extracted {len(gpt_texts)} non-empty texts from GPT data.")

    human_texts = []
    for entry in human_data:
        text_content = entry.get(text_key, "").strip()
        if text_content: # 确保文本不为空
            human_texts.append(text_content)
    print(f"Extracted {len(human_texts)} non-empty texts from Human data.")

    # --- 3. 划分训练集和测试集 ---
    random.seed(42) # 保证划分结果可复现
    random.shuffle(gpt_texts)
    random.shuffle(human_texts)

    gpt_train_count = int(len(gpt_texts) * train_ratio)
    human_train_count = int(len(human_texts) * train_ratio)

    gpt_train_texts = gpt_texts[:gpt_train_count]
    gpt_test_texts = gpt_texts[gpt_train_count:]

    human_train_texts = human_texts[:human_train_count]
    human_test_texts = human_texts[human_train_count:]

    print(f"GPT Data: Train {len(gpt_train_texts)}, Test {len(gpt_test_texts)}")
    print(f"Human Data: Train {len(human_train_texts)}, Test {len(human_test_texts)}")

    # --- 4. 创建目标目录结构 ---
    train_human_dir = os.path.join(output_root_dir, "train", "human")
    train_machine_dir = os.path.join(output_root_dir, "train", "machine")
    test_dir = os.path.join(output_root_dir, "test")

    os.makedirs(train_human_dir, exist_ok=True)
    os.makedirs(train_machine_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    print(f"Created output directories under: {output_root_dir}")

    # --- 5. 保存文本文件 ---
    # 保存训练集
    print("Saving training data...")
    for i, text in enumerate(human_train_texts):
        with open(os.path.join(train_human_dir, f"doc_human_{i}.txt"), 'w', encoding='utf-8') as f:
            f.write(text)
    for i, text in enumerate(gpt_train_texts):
        with open(os.path.join(train_machine_dir, f"doc_machine_{i}.txt"), 'w', encoding='utf-8') as f:
            f.write(text)
    print("Training data saved.")

    # 保存测试集
    print("Saving testing data...")
    for i, text in enumerate(human_test_texts):
        with open(os.path.join(test_dir, f"doc_test_human_{i}.txt"), 'w', encoding='utf-8') as f:
            f.write(text)
    for i, text in enumerate(gpt_test_texts):
        with open(os.path.join(test_dir, f"doc_test_machine_{i}.txt"), 'w', encoding='utf-8') as f:
            f.write(text)
    print("Testing data saved.")

    print("\nDataset preparation complete!")
    print(f"Generated dataset in: {os.path.abspath(output_root_dir)}")
    print("You can now use this dataset with feature_ref_generater.py and run_evaluation.py.")

if __name__ == "__main__":
    # 示例用法：
    # 假设你的 essay_GPT.json 和 essay_human.json 在当前目录下
    gpt_json_file = "./dataset/Yelp Review/yelp_GPT_concise.json"
    human_json_file = "./dataset/Yelp Review/yelp_human.json"
    output_dataset_root = "./my_yelp_dataset" # 为新的数据集指定一个不同的输出目录

    prepare_essay_dataset(
        gpt_json_file,
        human_json_file,
        output_dataset_root,
        train_ratio=0.8,
        text_key="Text" # 明确指定文本键为 "Text"
    )

    # 如果你希望保留 prepare_code_dataset 函数，可以将其重命名，
    # 然后再定义一个 prepare_essay_dataset 函数，或者使其更通用。
    # 为了清晰起见，我直接将原函数改为适应新数据。
    # 如果你希望同时处理两种数据，你可以复制这个文件并改名，或者将函数参数化得更彻底。
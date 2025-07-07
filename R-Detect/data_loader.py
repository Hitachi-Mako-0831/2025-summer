import random
import tqdm
import datasets
import re
import transformers
import numpy as np
from utils import MGT, HWT, config

preproc_tokenizer = transformers.AutoTokenizer.from_pretrained(
    "google-t5/t5-small", model_max_length=512
)


def process_spaces(text):
    text = (
        text.replace(" ,", ",")
        .replace(" .", ".")
        .replace(" ?", "?")
        .replace(" !", "!")
        .replace(" ;", ";")
        .replace(" '", "'")
        .replace(" ’ ", "'")
        .replace(" :", ":")
        .replace("<newline>", "\n")
        .replace("`` ", '"')
        .replace(" ''", '"')
        .replace("''", '"')
        .replace(".. ", "... ")
        .replace(" )", ")")
        .replace("( ", "(")
        .replace(" n't", "n't")
        .replace(" i ", " I ")
        .replace(" i'", " I'")
        .replace("\\'", "'")
        .replace("\n ", "\n")
        .strip()
    )
    text = text.replace("\r\n", "\n").replace("\\n", "").replace("!\n", "")
    return re.sub("\n+", "\n", text)


def trim_to_shorter_length(texta, textb):
    # truncate to shorter of o and s
    shorter_length = min(len(texta.split(" ")), len(textb.split(" ")))
    texta = " ".join(texta.split(" ")[:shorter_length])
    textb = " ".join(textb.split(" ")[:shorter_length])
    return texta, textb


import datasets
import tqdm
# 假设 HWT, MGT, process_spaces 和 config 已经定义在其他地方或作为参数传入

def load_HC3():
    # 检查config中是否有本地数据集路径
    if config["local_dataset"]:
        local_data_path = config["local_dataset"]
        print(f"Loading local HC3 dataset from: {local_data_path}")
        try:
            # 尝试从本地文件加载。假设你的HC3数据集是all.jsonl文件
            # 如果你的all.jsonl在 config["local_dataset"] 目录下，则路径为 f"{local_data_path}/all.jsonl"
            # 如果 all.jsonl 就是 config["local_dataset"] 本身，则直接用 local_data_path
            
            # 假设 all.jsonl 文件在 config["local_dataset"] 指定的目录下
            # 请根据你的实际文件路径调整
            jsonl_file_path = f"{local_data_path}/all.jsonl" 
            
            # 使用 "json" 类型加载jsonl文件
            ds = datasets.load_dataset("json", data_files=jsonl_file_path)
            
            # 由于HC3数据集的原始结构，它可能没有直接的"name"参数，或者需要调整
            # 如果加载json文件后没有"train"键，请检查数据集结构
            if "train" in ds:
                ds = ds["train"]  # DatasetDict -> Dataset
            else:
                # 如果没有 'train' 分割，直接使用加载的数据集对象
                # 这取决于你的all.jsonl文件是如何组织的
                ds = ds 

        except Exception as e:
            print(f"Error loading local dataset from {jsonl_file_path}: {e}")
            print("Falling back to remote HC3 dataset download.")
            # 如果本地加载失败，则回退到远程下载
            ds = datasets.load_dataset("Hello-SimpleAI/HC3", name="all")
            ds = ds["train"] # 远程下载通常会有train split
    else:
        print("Loading remote HC3 dataset")
        ds = datasets.load_dataset("Hello-SimpleAI/HC3", name="all")
        ds = ds["train"]  # DatasetDict -> Dataset

    filtered_ds = [
        item
        for item in ds
        if (
            len(item.get("human_answers", [])) > 0 # 使用.get避免keyError
            and len(item.get("chatgpt_answers", [])) > 0
            and len(item.get("human_answers", [])[0].split()) > 5 if item.get("human_answers") else False # 确保列表不为空才split
            and len(item.get("chatgpt_answers", [])[0].split()) > 5 if item.get("chatgpt_answers") else False
        )
    ]
    # print("DEBUG: filtered_ds[0]:", filtered_ds[0])

    data_new = {"text": [], "label": []}

    for i in tqdm.tqdm(range(len(filtered_ds)), desc="Parsing data"):
        # 再次检查列表是否为空或元素是否存在
        if filtered_ds[i].get("human_answers") and filtered_ds[i]["human_answers"][0]:
            data_new["text"].append(process_spaces(filtered_ds[i]["human_answers"][0]))
            data_new["label"].append(HWT)
        
        if filtered_ds[i].get("chatgpt_answers") and filtered_ds[i]["chatgpt_answers"][0]:
            data_new["text"].append(process_spaces(filtered_ds[i]["chatgpt_answers"][0]))
            data_new["label"].append(MGT)
            
    return data_new


def filter_data(data_o, long_train_threshold_low=150, long_train_threshold_high=512):
    data_HWT = [
        text for text, label in zip(data_o["text"], data_o["label"]) if label == HWT
    ]
    data_MGT = [
        text for text, label in zip(data_o["text"], data_o["label"]) if label == MGT
    ]

    # keep only examples with <= 512 tokens according to mask_tokenizer
    # this step has the extra effect of removing examples with low-quality/garbage content
    tokenized_data = preproc_tokenizer(data_HWT)
    long_HWT = [
        x
        for x, y in zip(data_HWT, tokenized_data["input_ids"])
        if long_train_threshold_low <= len(y) <= long_train_threshold_high
    ]
    tokenized_data = preproc_tokenizer(data_MGT)
    long_MGT = [
        x
        for x, y in zip(data_MGT, tokenized_data["input_ids"])
        if long_train_threshold_low <= len(y) <= long_train_threshold_high
    ]

    # print stats about remainining data
    print(f"Total number of samples: {len(long_HWT)}")
    print(f"Average number of words: {np.mean([len(x.split()) for x in long_HWT])}")

    data = {
        HWT: [],
        MGT: [],
    }

    # print(len(long_HWT), len(long_MGT))
    for o, s in zip(long_HWT, long_MGT):
        o, s = trim_to_shorter_length(o, s)

        # add to the data
        data[HWT].append(o)
        data[MGT].append(s)

    return data


# Test code
# data_o = load_HC3()
# data = filter_data(data_o)
# real = data[HWT]  # [:args.train_real_num]  len== n_samples, many sentences of words
# generated = data[MGT]
# print(real[:5])
# print(generated[:5])
# data_loader.py


import os

# 注意：这里不再需要从 utils 导入 config，因为我们会直接传递路径
# feature_ref_generater.py 会从 config 中获取 local_dataset 路径并传给我们

def load_data_from_folders(dataset_root_path):
    """
    根据给定的根路径，加载 'human' 和 'machine' 子目录中的文本文件。
    返回一个字典，键为 'HWT' 和 'MGT'，值是文本列表。
    """
    print(f"Loading raw data from: {dataset_root_path}")
    human_texts = []
    machine_texts = []

    # 尝试加载训练集中的human数据
    human_train_dir = os.path.join(dataset_root_path, "train", "human")
    if os.path.exists(human_train_dir):
        for filename in os.listdir(human_train_dir):
            if filename.endswith(".txt"):
                filepath = os.path.join(human_train_dir, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        human_texts.append(f.read())
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
        print(f"Loaded {len(human_texts)} human texts for HWT reference from {human_train_dir}")
    else:
        print(f"Warning: Human training data directory not found: {human_train_dir}")


    # 尝试加载训练集中的machine数据
    machine_train_dir = os.path.join(dataset_root_path, "train", "machine")
    if os.path.exists(machine_train_dir):
        for filename in os.listdir(machine_train_dir):
            if filename.endswith(".txt"):
                filepath = os.path.join(machine_train_dir, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        machine_texts.append(f.read())
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
        print(f"Loaded {len(machine_texts)} machine texts for MGT reference from {machine_train_dir}")
    else:
        print(f"Warning: Machine training data directory not found: {machine_train_dir}")

    # 返回一个字典，键为 'HWT' 和 'MGT'，值是文本列表
    return {"HWT": human_texts, "MGT": machine_texts}


def filter_my_data(data_dict):
    """
    这是一个示例过滤函数。你可以根据需要在此处添加更复杂的过滤逻辑。
    目前，它只是移除了空文本。
    """
    print("Applying basic data filtering (removing empty texts).")
    
    filtered_hwt = [text for text in data_dict['HWT'] if text.strip()]
    filtered_mgt = [text for text in data_dict['MGT'] if text.strip()]
    
    if len(filtered_hwt) < len(data_dict['HWT']):
        print(f"Removed {len(data_dict['HWT']) - len(filtered_hwt)} empty HWT texts.")
    if len(filtered_mgt) < len(data_dict['MGT']):
        print(f"Removed {len(data_dict['MGT']) - len(filtered_mgt)} empty MGT texts.")

    return {"HWT": filtered_hwt, "MGT": filtered_mgt}

# 为了兼容 feature_ref_generater.py 中的原始调用方式，
# 我们将旧函数名映射到新函数名
load_HC3 = load_data_from_folders
filter_data = filter_my_data
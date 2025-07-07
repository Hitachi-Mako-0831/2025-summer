import random
import datasets
import os
import json

SEPARATOR = '<<<SEP>>>'


DATASETS = ['writing', 'english', 'german', 'pubmed', 'my_custom_local'] # <--- 添加你的数据集名称



def load_pubmed(cache_dir):
    data = datasets.load_dataset('pubmed_qa', 'pqa_labeled', split='train', cache_dir=cache_dir)
    
    # combine question and long_answer
    data = [f'Question: {q} Answer:{SEPARATOR}{a}' for q, a in zip(data['question'], data['long_answer'])]

    return data


def process_prompt(prompt):
    return prompt.replace('[ WP ]', '').replace('[ OT ]', '')


def process_spaces(story):
    return story.replace(
        ' ,', ',').replace(
        ' .', '.').replace(
        ' ?', '?').replace(
        ' !', '!').replace(
        ' ;', ';').replace(
        ' \'', '\'').replace(
        ' ’ ', '\'').replace(
        ' :', ':').replace(
        '<newline>', '\n').replace(
        '`` ', '"').replace(
        ' \'\'', '"').replace(
        '\'\'', '"').replace(
        '.. ', '... ').replace(
        ' )', ')').replace(
        '( ', '(').replace(
        ' n\'t', 'n\'t').replace(
        ' i ', ' I ').replace(
        ' i\'', ' I\'').replace(
        '\\\'', '\'').replace(
        '\n ', '\n').strip()


def load_writing(cache_dir=None):
    writing_path = 'data/writingPrompts'
    
    with open(f'{writing_path}/valid.wp_source', 'r') as f:
        prompts = f.readlines()
    with open(f'{writing_path}/valid.wp_target', 'r') as f:
        stories = f.readlines()
    
    prompts = [process_prompt(prompt) for prompt in prompts]
    joined = [process_spaces(prompt + " " + story) for prompt, story in zip(prompts, stories)]
    filtered = [story for story in joined if 'nsfw' not in story and 'NSFW' not in story]

    random.seed(0)
    random.shuffle(filtered)

    return filtered


def load_language(language, cache_dir):
    # load either the english or german portion of the wmt16 dataset
    assert language in ['en', 'de']
    d = datasets.load_dataset('wmt16', 'de-en', split='train', cache_dir=cache_dir)
    docs = d['translation']
    desired_language_docs = [d[language] for d in docs]
    lens = [len(d.split()) for d in desired_language_docs]
    sub = [d for d, l in zip(desired_language_docs, lens) if l > 100 and l < 150]
    return sub


def load_german(cache_dir):
    return load_language('de', cache_dir)


def load_english(cache_dir):
    return load_language('en', cache_dir)

def load_my_custom_local(cache_dir=None):
    """
    加载本地的 Index, Text, Source 格式的 JSON 数据集。
    假设文件路径是 project_root/dataset/my_custom_local_data.json
    """

    local_json_file_path = os.path.join('dataset', 'News', 'news_combine.json')
    
    print(f"尝试从本地加载数据集: {local_json_file_path}")

    if not os.path.exists(local_json_file_path):
        raise FileNotFoundError(f"未找到本地数据集文件：{local_json_file_path}。请检查路径是否正确。")

    # 使用 datasets 库加载本地 JSON 文件
    # 它会自动解析 Index, Text, Source 字段
    loaded_data = datasets.load_dataset('json', data_files=local_json_file_path, split='train')

    original_texts = [] # 用于存储 Source 为 "Human" 的文本
    sampled_texts = []  # 用于存储 Source 为 "GPT" 的文本 (或其他 AI 来源)

    # 遍历加载的数据集，根据 'Source' 字段进行分类
    for item in loaded_data:
        text = item['Text']
        source = item['Source']

        # 进行基本的文本清理，与 generate_data 中的步骤保持一致
        cleaned_text = process_spaces(text) # 移除换行符
        cleaned_text = cleaned_text.strip() # 移除首尾空白

        # 如果文本为空，则跳过
        if not cleaned_text:
            continue

        if source == "human":
            original_texts.append(cleaned_text)
        elif source == "GPT": # 或者根据你的实际数据中的 AI 来源标签来判断
            sampled_texts.append(cleaned_text)
        # 你可以添加更多的 else if 来处理其他类型的 Source，如果需要

    # 进一步处理：去重、长度筛选和随机打乱（可选，但推荐与 generate_data 行为保持一致）
    # 通常 generate_data 还会进行去重和长度筛选
    # 这里我们只做最基本的分类，你可以在 generate_data 中让这些通用处理生效

    # 将结果打包成 eval_supervised 所需的格式
    formatted_data = {
        'original': original_texts[:400],
        'sampled': sampled_texts[:400]
    }
    
    print(f"已加载 {len(original_texts)} 条 'original' (Human) 文本。")
    print(f"已加载 {len(sampled_texts)} 条 'sampled' (GPT) 文本。")

    return formatted_data

def load(name, cache_dir, **kwargs):
    if name in DATASETS:
        load_fn = globals()[f'load_{name}']
        return load_fn(cache_dir=cache_dir, **kwargs)
    else:
        raise ValueError(f'Unknown dataset {name}')
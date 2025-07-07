import numpy as np
import dill as pickle
import tiktoken
import openai
import argparse
import os
import json # 导入 json 模块
from sklearn.metrics import roc_auc_score # 导入 AUROC 计算函数

from sklearn.linear_model import LogisticRegression
from utils.featurize import normalize, t_featurize_logprobs, score_ngram
from utils.symbolic import train_trigram, get_words, vec_functions, scalar_functions

openai.base_url = "https://api.openai-hk.com/v1/"
openai.api_key = '-'


# --- 参数解析 ---
parser = argparse.ArgumentParser(description="Classify text from a JSON dataset and calculate AUROC.")
parser.add_argument("--json_file", type=str, default="dataset/News/news_combine.json",
                    help="Path to the JSON dataset file containing 'Index', 'Text', and 'Source'.")
parser.add_argument("--openai_key", type=str, default="",
                    help="Your OpenAI API key. Can also be set as OPENAI_API_KEY environment variable.")
args = parser.parse_args()

# --- 常量和模型加载 ---
MAX_TOKENS = 4000 # 针对 gpt-3.5-turbo 的上下文限制
best_features = open("model/features.txt").read().strip().split("\n")
enc = tiktoken.get_encoding("cl100k_base") 

model = pickle.load(open("model/model", "rb"))
mu = pickle.load(open("model/mu", "rb"))
sigma = pickle.load(open("model/sigma", "rb"))

# --- N-gram 模型加载 ---
print("Loading Trigram Model...")
trigram_model = train_trigram()

# --- 辅助函数：从 ChatCompletion 模型获取 logprobs ---
def get_logprobs_from_chat_model(input_doc_content, target_model, expected_len_tokens):
    """
    辅助函数：从 ChatCompletion 模型获取指定文档的 logprobs。
    """
    gpt_token_logprobs = []
    gpt_subwords = []
    try:
        completion = openai.chat.completions.create(
            model=target_model,
            messages=[
                {
                    "role": "user",
                    "content": input_doc_content,
                },
            ],
            logprobs=True,
            top_logprobs=1,
            temperature=0.0 # 保持确定性，使 logprobs 更稳定
        )

        if completion.choices[0].logprobs and completion.choices[0].logprobs.content:
            for item in completion.choices[0].logprobs.content:
                gpt_token_logprobs.append(np.exp(item.logprob))
                gpt_subwords.append(item.token)
            
            # 裁剪或填充以匹配预期长度
            if len(gpt_token_logprobs) > expected_len_tokens:
                gpt_token_logprobs = gpt_token_logprobs[:expected_len_tokens]
                gpt_subwords = gpt_subwords[:expected_len_tokens]
            elif len(gpt_token_logprobs) < expected_len_tokens:
                padding_needed = expected_len_tokens - len(gpt_token_logprobs)
                gpt_token_logprobs.extend([0.0] * padding_needed)
                gpt_subwords.extend(["<UNK>"] * padding_needed)
        else:
            print(f"警告: {target_model} 未能为文本返回有效的 logprobs 内容。将填充默认值。")
            gpt_token_logprobs = [0.0] * expected_len_tokens
            gpt_subwords = [enc.decode([t]) for t in enc.encode(input_doc_content)[:expected_len_tokens]]

    except openai.APIConnectionError as e:
        print(f"API连接错误 ({target_model}): {e}。将填充默认值。")
        gpt_token_logprobs = [0.0] * expected_len_tokens
        gpt_subwords = [enc.decode([t]) for t in enc.encode(input_doc_content)[:expected_len_tokens]]
    except openai.APIStatusError as e:
        print(f"API状态错误 ({target_model}) {e.status_code}: {e.response}。将填充默认值。")
        gpt_token_logprobs = [0.0] * expected_len_tokens
        gpt_subwords = [enc.decode([t]) for t in enc.encode(input_doc_content)[:expected_len_tokens]]
    except openai.AuthenticationError as e:
        print(f"API认证错误 ({target_model}): {e}。将填充默认值。")
        gpt_token_logprobs = [0.0] * expected_len_tokens
        gpt_subwords = [enc.decode([t]) for t in enc.encode(input_doc_content)[:expected_len_tokens]]
    except Exception as e:
        print(f"调用 {target_model} API 时发生意外错误: {e}。将填充默认值。")
        gpt_token_logprobs = [0.0] * expected_len_tokens
        gpt_subwords = [enc.decode([t]) for t in enc.encode(input_doc_content)[:expected_len_tokens]]
        
    return np.array(gpt_token_logprobs), gpt_subwords

# --- 主处理逻辑 ---
json_file_path = args.json_file
all_predictions = [] # 存储所有预测概率
all_true_labels = [] # 存储所有真实标签
processed_results = [] # 存储每个条目的详细结果

if not os.path.exists(json_file_path):
    print(f"错误: JSON 文件 '{json_file_path}' 不存在。")
    exit(1)

print(f"Processing dataset from: {json_file_path}")

try:
    with open(json_file_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
except json.JSONDecodeError as e:
    print(f"错误: 无法解析 JSON 文件 '{json_file_path}': {e}")
    exit(1)
except Exception as e:
    print(f"读取文件 '{json_file_path}' 时发生错误: {e}")
    exit(1)

print(f"Total entries to process: {len(dataset)}")

for entry in dataset:
    index = entry.get("Index")
    doc = entry.get("Text")
    source = entry.get("Source")

    if doc is None or source is None:
        print(f"警告: 跳过索引 {index} 的条目，因为它缺少 'Text' 或 'Source' 字段。")
        continue

    # 映射 Source 到数值标签
    true_label = 1 if source.upper() == "GPT" else 0 # GPT 为 1，human 为 0

    print(f"\n--- Processing Index: {index} (Source: {source}) ---")
    print(f"Input snippet for Index {index}: {doc[:200]}...") # 打印部分内容方便查看

    tokens_ids = enc.encode(doc)
    if len(tokens_ids) > MAX_TOKENS:
        tokens_ids = tokens_ids[:MAX_TOKENS]
        doc_truncated = enc.decode(tokens_ids).strip()
        print(f"警告：文本索引 {index} 已截断至 {MAX_TOKENS} tokens，以适应 gpt-3.5-turbo 的上下文限制。")
    else:
        doc_truncated = doc # 如果没有截断，就用原始 doc

    # --- 特征提取 ---
    trigram = np.array(score_ngram(doc_truncated, trigram_model, enc.encode, n=3, strip_first=False))
    unigram = np.array(score_ngram(doc_truncated, trigram_model.base, enc.encode, n=1, strip_first=False))

    expected_doc_tokens_count = len(enc.encode(doc_truncated))

    # 模拟 Ada (使用 gpt-3.5-turbo)
    print("Fetching Ada-like logprobs using gpt-3.5-turbo-1106...")
    ada_logprobs_new, ada_subwords = get_logprobs_from_chat_model(
        doc_truncated, "gpt-3.5-turbo-1106", expected_doc_tokens_count
    )

    # 模拟 Davinci (使用 gpt-3.5-turbo-16k)
    print("Fetching Davinci-like logprobs using gpt-3.5-turbo-16k...")
    davinci_logprobs_new, davinci_subwords = get_logprobs_from_chat_model(
        doc_truncated, "gpt-3.5-turbo-16k", expected_doc_tokens_count
    )

    subwords = ada_subwords # 使用 Ada 的 subwords，因为它们应该是一致的
    gpt2_map = {"\n": "Ċ", "\t": "ĉ", " ": "Ġ"}
    for i in range(len(subwords)):
        for k, v in gpt2_map.items():
            subwords[i] = subwords[i].replace(k, v)

    t_features = t_featurize_logprobs(davinci_logprobs_new, ada_logprobs_new, subwords)

    vector_map = {
        "davinci-logprobs": davinci_logprobs_new,
        "ada-logprobs": ada_logprobs_new,
        "trigram-logprobs": trigram,
        "unigram-logprobs": unigram
    }

    exp_features = []
    for exp in best_features:
        exp_tokens = get_words(exp)
        curr_vec_name = exp_tokens[0]

        try:
            curr = vector_map[curr_vec_name]
        except KeyError:
            print(f"警告：无法找到特征表达式 '{exp}' 中的初始向量 '{curr_vec_name}'，跳过此特征。")
            continue

        for i in range(1, len(exp_tokens)):
            token = exp_tokens[i]
            if token in vec_functions:
                if i + 1 >= len(exp_tokens):
                    print(f"警告：特征表达式 '{exp}' 格式错误或不完整，跳过。")
                    break
                next_vec_name = exp_tokens[i+1]
                try:
                    next_vec = vector_map[next_vec_name]
                except KeyError:
                    print(f"警告：无法找到特征表达式 '{exp}' 中的后续向量 '{next_vec_name}'，跳过此特征。")
                    break
                curr = vec_functions[token](curr, next_vec)
                i += 1
            elif token in scalar_functions:
                exp_features.append(scalar_functions[token](curr))
                break
    
    # --- 预测 ---
    data = (np.array(t_features + exp_features) - mu) / sigma
    pred_prob = model.predict_proba(data.reshape(-1, 1).T)[:, 1][0] # 获取 AI 生成的概率

    all_predictions.append(pred_prob)
    all_true_labels.append(true_label)


    processed_results.append({
        "Index": index,
        "OriginalText": entry.get("Text"), # 保存原始完整文本
        "TruncatedText": doc_truncated, # 保存截断后的文本
        "Source": source,
        "TrueLabel": true_label,
        "PredictedProbability_AI": float(pred_prob), # 转换为 float 以便 JSON 序列化
    })

# --- 计算 AUROC 并保存结果 ---
print("\n--- All Processing Complete ---")

if len(all_predictions) > 1 and len(np.unique(all_true_labels)) > 1:
    # 确保至少有两个样本且真实标签有不同类别才能计算 AUROC
    try:
        auroc_score = roc_auc_score(all_true_labels, all_predictions)
        print(f"Calculated AUROC for the dataset: {auroc_score:.4f}")
    except ValueError as e:
        print(f"无法计算 AUROC: {e}. 这通常意味着数据集中只有一个类别的样本，或者样本数量不足。")
        auroc_score = None
else:
    print("样本数量不足或真实标签只有单一类别，无法计算 AUROC。")
    auroc_score = None

# 保存结果到新的 JSON 文件
output_json_filename = os.path.splitext(os.path.basename(json_file_path))[0] + "_classified_results_news.json"
output_data = {
    "dataset_source_file": os.path.basename(json_file_path),
    "total_entries_processed": len(processed_results),
    "auroc_score": auroc_score,
    "predictions": processed_results
}

with open(output_json_filename, 'w', encoding='utf-8') as outfile:
    json.dump(output_data, outfile, indent=4, ensure_ascii=False)

print(f"All results saved to: {output_json_filename}")
print("\nTest complete. Remember that the current model is trained on older OpenAI models.")
print("For best results with gpt-3.5-turbo-1106 and gpt-3.5-turbo-16k, consider retraining the model.")
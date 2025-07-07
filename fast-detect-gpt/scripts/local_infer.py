# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random

import numpy as np
import torch
import os
import glob
import argparse
import json
import tqdm
from model import load_tokenizer, load_model
from fast_detect_gpt import get_sampling_discrepancy_analytic
from metrics import get_roc_metrics, get_precision_recall_metrics # Assuming these are available from your previous scripts
from scipy.stats import norm


# Considering balanced classification that p(D0) equals to p(D1), we have
#   p(D1|x) = p(x|D1) / (p(x|D1) + p(x|D0))
def compute_prob_norm(x, mu0, sigma0, mu1, sigma1):
    pdf_value0 = norm.pdf(x, loc=mu0, scale=sigma0)
    pdf_value1 = norm.pdf(x, loc=mu1, scale=sigma1)
    prob = pdf_value1 / (pdf_value0 + pdf_value1)
    return prob

class FastDetectGPT:
    def __init__(self, args):
        self.args = args
        self.criterion_fn = get_sampling_discrepancy_analytic
        self.scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.cache_dir)
        self.scoring_model = load_model(args.scoring_model_name, args.device, args.cache_dir)
        self.scoring_model.eval()
        if args.sampling_model_name != args.scoring_model_name:
            self.sampling_tokenizer = load_tokenizer(args.sampling_model_name, args.cache_dir)
            self.sampling_model = load_model(args.sampling_model_name, args.device, args.cache_dir)
            self.sampling_model.eval()
        # To obtain probability values that are easy for users to understand, we assume normal distributions
        # of the criteria and statistic the parameters on a group of dev samples. The normal distributions are defined
        # by mu0 and sigma0 for human texts and by mu1 and sigma1 for AI texts. We set sigma1 = 2 * sigma0 to
        # make sure of a wider coverage of potential AI texts.
        # Note: the probability could be high on both left side and right side of Normal(mu0, sigma0).
        #   gpt-j-6B_gpt-neo-2.7B: mu0: 0.2713, sigma0: 0.9366, mu1: 2.2334, sigma1: 1.8731, acc:0.8122
        #   gpt-neo-2.7B_gpt-neo-2.7B: mu0: -0.2489, sigma0: 0.9968, mu1: 1.8983, sigma1: 1.9935, acc:0.8222
        #   falcon-7b_falcon-7b-instruct: mu0: -0.0707, sigma0: 0.9520, mu1: 2.9306, sigma1: 1.9039, acc:0.8938
        distrib_params = {
            'gpt-j-6B_gpt-neo-2.7B': {'mu0': 0.2713, 'sigma0': 0.9366, 'mu1': 2.2334, 'sigma1': 1.8731},
            'gpt-neo-2.7B_gpt-neo-2.7B': {'mu0': -0.2489, 'sigma0': 0.9968, 'mu1': 1.8983, 'sigma1': 1.9935},
            'falcon-7b_falcon-7b-instruct': {'mu0': -0.0707, 'sigma0': 0.9520, 'mu1': 2.9306, 'sigma1': 1.9039},
        }
        key = f'{args.sampling_model_name}_{args.scoring_model_name}'
        self.classifier = distrib_params[key]

    # compute conditional probability curvature
    def compute_crit(self, text):
        tokenized = self.scoring_tokenizer(text, truncation=True, return_tensors="pt", padding=True, return_token_type_ids=False).to(self.args.device)
        labels = tokenized.input_ids[:, 1:]
        with torch.no_grad():
            logits_score = self.scoring_model(**tokenized).logits[:, :-1]
            if self.args.sampling_model_name == self.args.scoring_model_name:
                logits_ref = logits_score
            else:
                tokenized = self.sampling_tokenizer(text, truncation=True, return_tensors="pt", padding=True, return_token_type_ids=False).to(self.args.device)
                assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = self.sampling_model(**tokenized).logits[:, :-1]
            crit = self.criterion_fn(logits_ref, logits_score, labels)
        return crit, labels.size(1)

    # compute probability
    def compute_prob(self, text):
        crit, ntoken = self.compute_crit(text)
        mu0 = self.classifier['mu0']
        sigma0 = self.classifier['sigma0']
        mu1 = self.classifier['mu1']
        sigma1 = self.classifier['sigma1']
        prob = compute_prob_norm(crit, mu0, sigma0, mu1, sigma1)
        return prob, crit, ntoken


# run interactive local inference
# def run(args):
#     detector = FastDetectGPT(args)
#     # input text
#     print('Local demo for Fast-DetectGPT, where the longer text has more reliable result.')
#     print('')
#     while True:
#         print("Please enter your text: (Press Enter twice to start processing)")
#         lines = []
#         while True:
#             line = input()
#             if len(line) == 0:
#                 break
#             lines.append(line)
#         text = "\n".join(lines)
#         if len(text) == 0:
#             break
#         # estimate the probability of machine generated text
#         prob, crit, ntokens = detector.compute_prob(text)
#         print(f'Fast-DetectGPT criterion is {crit:.4f}, suggesting that the text has a probability of {prob * 100:.0f}% to be machine-generated.')
#         print()

def run(args):
    detector = FastDetectGPT(args)
    
    # 确保提供了数据集文件路径
    if not args.dataset_file:
        print("错误：请通过 --dataset_file 参数提供数据集 JSON 文件的路径。")
        return

    print(f"正在加载数据集：{args.dataset_file}...")
    try:
        with open(args.dataset_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"错误：未找到数据集文件：{args.dataset_file}")
        return
    except json.JSONDecodeError:
        print(f"错误：无法解析数据集文件 {args.dataset_file}。请确保它是有效的 JSON 格式。")
        return
    
    print(f"已加载 {len(dataset)} 条记录。开始处理...")

    results = []
    true_labels = [] # 用于存储真实标签 (0 for human, 1 for AI)
    predictions_crit = [] # 用于存储模型预测的crit值

    # 遍历数据集中的每条记录
    # tqdm 提供了一个进度条，方便查看处理进度
    for item in tqdm.tqdm(dataset, desc="正在处理文本"):
        text_id = item.get("Index") # 获取索引，如果没有则为 None
        text_content = item.get("Text")
        source_label = item.get("Source") # 获取原始来源标签

        if not text_content:
            print(f"警告：记录 {text_id if text_id is not None else '未知'} 缺少 'Text' 字段，跳过。")
            continue
        
        # 估算文本为机器生成的概率和判别分数
        prob, crit, ntokens = detector.compute_prob(text_content)
        
        # 收集真实标签 (将 'human' 映射为 0，'GPT'/'AI' 映射为 1)
        # 假设 Source 字段只有 'human' 或 'GPT'/'AI'
        if source_label and source_label.lower() == 'human':
            true_labels.append(0)
        elif source_label and (source_label.lower() == 'gpt' or source_label.lower() == 'ai'):
            true_labels.append(1)
        else:
            # 如果 Source 字段格式不符合预期，可以决定跳过或给出警告
            print(f"警告：记录 {text_id} 的 'Source' 字段 '{source_label}' 未知，跳过。")
            continue
            
        predictions_crit.append(crit)

        # 保存每条记录的详细结果
        results.append({
            "Index": text_id,
            "Text": text_content,
            "OriginalSource": source_label,
            "DetectedCrit": float(crit), # crit 是 tensor，转换为 float
            "DetectedProbability": float(prob), # prob 是 tensor，转换为 float
            "NumTokens": ntokens
        })
    
    # 确保收集到了足够的样本来计算 AUROC
    if len(true_labels) < 2 or len(np.unique(true_labels)) < 2:
        print("警告：数据集中标签类别不足或样本数量太少，无法计算 AUROC。")
        roc_auc = None
        pr_auc = None
    else:
        # 计算 AUROC 和 PR-AUC
        # get_roc_metrics 函数期望输入两个列表：真实样本的得分和机器样本的得分
        # 因此我们需要根据 true_labels 分离 crit 预测
        real_crits = [predictions_crit[i] for i, label in enumerate(true_labels) if label == 0]
        fake_crits = [predictions_crit[i] for i, label in enumerate(true_labels) if label == 1]
        
        fpr, tpr, roc_auc = get_roc_metrics(real_crits, fake_crits)
        p, r, pr_auc = get_precision_recall_metrics(real_crits, fake_crits)

        print(f"\n数据集总 AUROC: {roc_auc:.4f}")
        print(f"数据集总 PR-AUC: {pr_auc:.4f}")

    # 将所有结果以及最终的 AUROC 保存到新的 JSON 文件中
    output_filename = args.output_file if args.output_file else "detection_results.json"
    
    final_output = {
        "detection_details": results,
        "summary_metrics": {
            "total_samples_processed": len(results),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            # 可以添加其他您认为有用的统计信息
        }
    }

    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, ensure_ascii=False, indent=4)
        print(f"所有结果已保存到：{output_filename}")
    except IOError as e:
        print(f"错误：无法将结果保存到文件 {output_filename}：{e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sampling_model_name', type=str, default="gpt-neo-2.7B")
    parser.add_argument('--scoring_model_name', type=str, default="gpt-neo-2.7B")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    

    parser.add_argument('--dataset_file', type=str, 
                        help="要处理的 JSON 数据集文件的路径。",
                        default="dataset/News/news_combine.json")
    parser.add_argument('--output_file', type=str, 
                        help="保存检测结果的 JSON 文件路径，默认为 'detection_results.json'。",
                        default="detection_results_news.json")
    args = parser.parse_args()
    run(args)




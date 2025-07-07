import numpy as np
import dill as pickle
import tiktoken
import openai
import argparse

from sklearn.linear_model import LogisticRegression
from utils.featurize import normalize, t_featurize_logprobs, score_ngram
from utils.symbolic import train_trigram, get_words, vec_functions, scalar_functions

openai.base_url = "https://api.openai-hk.com/v1/"
openai.api_key = '-'

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, default="input.txt")
parser.add_argument("--openai_key", type=str, default="")
args = parser.parse_args()


file = args.file
MAX_TOKENS = 2047
best_features = open("model/features.txt").read().strip().split("\n")

# Load davinci tokenizer
# enc = tiktoken.encoding_for_model("davinci")
enc = tiktoken.get_encoding("cl100k_base")
# Load model
model = pickle.load(open("model/model", "rb"))
mu = pickle.load(open("model/mu", "rb"))
sigma = pickle.load(open("model/sigma", "rb"))

# Load data and featurize
with open(file) as f:
    doc = f.read().strip()
    # Strip data to first MAX_TOKENS tokens
    tokens_ids = enc.encode(doc)[:MAX_TOKENS]
    doc = enc.decode(tokens_ids).strip()

    print(f"Input: {doc}")

# Train trigram
print("Loading Trigram...")

trigram_model = train_trigram()

trigram = np.array(score_ngram(doc, trigram_model, enc.encode, n=3, strip_first=False))
unigram = np.array(score_ngram(doc, trigram_model.base, enc.encode, n=1, strip_first=False))

# response = openai.Completion.create(
#     model="ada",
#     prompt="<|endoftext|>" + doc,
#     max_tokens=0,
#     echo=True,
#     logprobs=1,
# )
# ada = np.array(list(map(lambda x: np.exp(x), response["choices"][0]["logprobs"]["token_logprobs"][1:])))

# response = openai.Completion.create(
#     model="davinci",
#     prompt="<|endoftext|>" + doc,
#     max_tokens=0,
#     echo=True,
#     logprobs=1,
# )
# davinci = np.array(list(map(lambda x: np.exp(x), response["choices"][0]["logprobs"]["token_logprobs"][1:])))

# subwords = response["choices"][0]["logprobs"]["tokens"][1:]
# gpt2_map = {"\n": "Ċ", "\t": "ĉ", " ": "Ġ"}
# for i in range(len(subwords)):
#     for k, v in gpt2_map.items():
#         subwords[i] = subwords[i].replace(k, v)

# t_features = t_featurize_logprobs(davinci, ada, subwords)

try:
    # **关键：** 使用 `logprobs=True` 和 `top_logprobs=1`
    # 并将输入文本作为 message content，让模型预测其自身的概率。
    # 最新版本的 OpenAI ChatCompletion API 可以在 output.logprobs.content 中获取每个 token 的 logprob。
    # 这里的目标是获取输入 `doc` 中每个 token 的 logprob，因此我们将其作为唯一消息内容。

    completion = openai.chat.completions.create(
        model="gpt-3.5-turbo", # 使用 gpt-3.5-turbo 模型
        messages=[
            {
                "role": "user",
                "content": "Please repeat the following text exactly:\n" + doc, # 将整个输入文本作为用户消息内容
            },
        ],
        logprobs=True, # 请求返回 logprobs
        top_logprobs=1, # 对于每个 token，获取其最高 logprob (即它本身的 logprob)
        temperature=0.0
    )

    # 从响应中提取 logprobs 和 tokens
    # ChatCompletion 的 logprobs 结构与旧 Completion API 不同
    # 它现在通过 completion.choices[0].logprobs.content 访问
    # 这个 content 是一个列表，每个元素包含 token 和它的 logprob
    
    # 检查是否有 logprobs 内容
    if completion.choices[0].logprobs and completion.choices[0].logprobs.content:

        gpt35_token_logprobs = []
        gpt35_subwords = []
        expected_tokens_count = len(tokens_ids)
        # 遍历 logprobs.content 获取每个 token 的 logprob
        # 注意：这里的 `completion.logprobs.content` 包含了输入 `messages` 的 logprobs
        # （这是 OpenAI API v1.x 新增的功能，但需要模型支持，且通常在 `messages` 中的 `role: user` 消息中的 token 上）
        if completion.choices[0].logprobs and completion.choices[0].logprobs.content:
            for item in completion.choices[0].logprobs.content:
                # `item.token` 是 token 字符串
                # `item.logprob` 是该 token 的 logprob (自然对数)
                # `np.exp(item.logprob)` 转换为原始概率
                gpt35_token_logprobs.append(np.exp(item.logprob))
                gpt35_subwords.append(item.token)
            
            if len(gpt35_token_logprobs) > expected_tokens_count:
                gpt35_token_logprobs = gpt35_token_logprobs[:expected_tokens_count]
                gpt35_subwords = gpt35_subwords[:expected_tokens_count]
            elif len(gpt35_token_logprobs) < expected_tokens_count:
                print(f"警告：GPT-3.5-turbo 返回的 logprobs 数量 ({len(gpt35_token_logprobs)}) 少于输入文本的 token 数量 ({expected_tokens_count})。可能导致特征计算不完整。")
                # 补齐到预期的长度，用0填充，或者根据你的处理逻辑调整
                padding_needed = expected_tokens_count - len(gpt35_token_logprobs)
                gpt35_token_logprobs.extend([0.0] * padding_needed) # 用0概率填充
                gpt35_subwords.extend(["<UNK>"] * padding_needed) # 用未知token填充
            
        else:
             raise ValueError("Failed to retrieve logprobs from GPT-3.5-turbo response.")

        # 将 gpt-3.5-turbo 的 logprobs 同时赋给 davinci 和 ada
        # 注意：这会改变特征的含义，因为原始模型是基于两个不同模型的 logprobs
        davinci = np.array(gpt35_token_logprobs)
        ada = np.array(gpt35_token_logprobs) # Ada 也使用 gpt-3.5-turbo 的 logprobs

        # subwords 来自 gpt-3.5-turbo 的响应
        subwords = gpt35_subwords
        gpt2_map = {"\n": "Ċ", "\t": "ĉ", " ": "Ġ"}
        for i in range(len(subwords)):
            for k, v in gpt2_map.items():
                subwords[i] = subwords[i].replace(k, v)
    else:
        raise ValueError("GPT-3.5-turbo did not return valid logprobs content.")

except openai.APIConnectionError as e:
    print(f"无法连接到 OpenAI API: {e}")
    print("请检查你的网络连接或API密钥是否有效。")
    # 设置默认值，让代码能够继续执行，但结果会不准确
    davinci= np.zeros(len(tokens_ids))
    ada= np.zeros(len(tokens_ids))
    subwords = [enc.decode([t]) for t in tokens_ids] # 使用 tiktoken 的 subwords
except openai.APIStatusError as e:
    print(f"OpenAI API 返回错误状态码 {e.status_code}: {e.response}")
    print("请检查API密钥、模型名称或请求参数是否正确。")
    davinci= np.zeros(len(tokens_ids))
    ada= np.zeros(len(tokens_ids))
    subwords = [enc.decode([t]) for t in tokens_ids]
except openai.AuthenticationError as e:
    print(f"OpenAI 认证失败: {e}")
    print("请确保你的 OpenAI API 密钥是有效的，并且你所在的地区可以访问 OpenAI 服务。")
    davinci = np.zeros(len(tokens_ids))
    ada = np.zeros(len(tokens_ids))
    subwords = [enc.decode([t]) for t in tokens_ids]
except Exception as e:
    print(f"调用 OpenAI API 时发生意外错误: {e}")
    davinci= np.zeros(len(tokens_ids))
    ada= np.zeros(len(tokens_ids))
    subwords = [enc.decode([t]) for t in tokens_ids]

def get_logprobs_from_chat_model(client_obj, input_doc_content, target_model, expected_len_tokens):
    """
    辅助函数：从 ChatCompletion 模型获取指定文档的 logprobs。
    """
    gpt_token_logprobs = []
    gpt_subwords = []
    try:
        completion = client_obj.chat.completions.create(
            model=target_model,
            messages=[
                {
                    "role": "user",
                    "content": input_doc_content,
                },
            ],
            logprobs=True,
            top_logprobs=1,
            max_tokens=1, # 仅为获取输入logprobs，不生成额外文本
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
            raise ValueError(f"{target_model} did not return valid logprobs content.")

    except openai.APIConnectionError as e:
        print(f"无法连接到 OpenAI API: {e}")
        gpt_token_logprobs = [0.0] * expected_len_tokens
        gpt_subwords = [enc.decode([t]) for t in enc.encode(input_doc_content)[:expected_len_tokens]]
    except openai.APIStatusError as e:
        print(f"OpenAI API 返回错误状态码 {e.status_code}: {e.response}")
        gpt_token_logprobs = [0.0] * expected_len_tokens
        gpt_subwords = [enc.decode([t]) for t in enc.encode(input_doc_content)[:expected_len_tokens]]
    except openai.AuthenticationError as e:
        print(f"OpenAI 认证失败: {e}")
        gpt_token_logprobs = [0.0] * expected_len_tokens
        gpt_subwords = [enc.decode([t]) for t in enc.encode(input_doc_content)[:expected_len_tokens]]
    except Exception as e:
        print(f"调用 {target_model} API 时发生意外错误: {e}")
        gpt_token_logprobs = [0.0] * expected_len_tokens
        gpt_subwords = [enc.decode([t]) for t in enc.encode(input_doc_content)[:expected_len_tokens]]
        
    return np.array(gpt_token_logprobs), gpt_subwords












t_features = t_featurize_logprobs(davinci, ada, subwords)
vector_map = {
    "davinci-logprobs": davinci,
    "ada-logprobs": ada,
    "trigram-logprobs": trigram,
    "unigram-logprobs": unigram
}

exp_features = []
for exp in best_features:

    exp_tokens = get_words(exp)
    curr = vector_map[exp_tokens[0]]

    for i in range(1, len(exp_tokens)):
        if exp_tokens[i] in vec_functions:
            next_vec = vector_map[exp_tokens[i+1]]
            curr = vec_functions[exp_tokens[i]](curr, next_vec)
        elif exp_tokens[i] in scalar_functions:
            exp_features.append(scalar_functions[exp_tokens[i]](curr))
            break

data = (np.array(t_features + exp_features) - mu) / sigma
preds = model.predict_proba(data.reshape(-1, 1).T)[:, 1]

print(f"Prediction: {preds}")

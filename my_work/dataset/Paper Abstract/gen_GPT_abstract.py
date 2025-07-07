import json
import openai
import time

# 设置 OpenAI API 密钥
# optional; defaults to `os.environ['OPENAI_API_KEY']`
openai.api_key = 'hk-p5ioqc10000518971d72d52ca5b76fcd3c56b2c5dac4a312'

# all client options can be configured just like the `OpenAI` instantiation counterpart
openai.base_url = "https://api.openai-hk.com/v1/"
# openai.default_headers = {"x-foo": "true"}
# 读取 paper_abstract.json 数据集
with open('prompt_for_gpt.json', 'r', encoding='utf-8') as file:
    paper_data = json.load(file)

# 定义生成文本的函数
def generate_gpt_summary(title, abstract):
    try:
        completion = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": f"The title is {title}, start with {abstract}, write a short concise abstract with roughly 175 words based on this: ",
                },
            ],
        )
        summary = completion.choices[0].message.content
        print(summary)
        return summary
    except Exception as e:
        print(f"错误：生成摘要时出错: {e}")
        return None

# 创建一个新的数据集
new_paper_data = []

# 遍历原数据集并调用 GPT 生成摘要
for item in paper_data:
    title = item["Title"]
    index = item["Index"]
    abstract = item["Start"]
    
    print(f"正在生成摘要：{index}")
    
    # 调用 GPT 生成摘要
    summary = generate_gpt_summary(title, abstract)
    
    if summary:
        new_paper_data.append({
            "Index": item["Index"],
            "Title": title,
            "Abstract": summary,  # 将生成的摘要添加到数据中
            "Source": "GPT"
        })
    
    # 控制请求频率，避免过快调用 API 被限制
    time.sleep(1)  # 等待 1 秒钟

# 将结果保存到 paper_abstract_GPT.json
with open('paper_abstract_GPT.json', 'w', encoding='utf-8') as outfile:
    json.dump(new_paper_data, outfile, ensure_ascii=False, indent=4)

print("数据已保存至 paper_abstract_GPT.json")


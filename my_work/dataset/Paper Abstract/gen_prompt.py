import json

# 读取 paper_abstract.json
with open('Paper_abstract.json', 'r', encoding='utf-8') as file:
    paper_data = json.load(file)

# 提取数据并重构格式
restructured_data = []
for item in paper_data:
    # 获取 Abstract 中的前 15 个单词
    abstract_words = item["Abstract"].split()
    start_text = " ".join(abstract_words[:15])  # 提取前 15 个单词
    
    # 重新构造数据格式
    restructured_data.append({
        "Index": item["Index"],
        "Title": item["Title"],
        "Abstract": item["Abstract"],
        "Source": "human"
    })

# 保存到 prompt_for_gpt.json
with open('Paper_abstract.json', 'w', encoding='utf-8') as outfile:
    json.dump(restructured_data, outfile, ensure_ascii=False, indent=4)

print("数据已保存至 prompt_for_gpt.json")

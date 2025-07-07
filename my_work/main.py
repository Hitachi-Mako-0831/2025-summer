import asyncio
from utils import *
import yaml

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.conditions import ExternalTermination, TextMentionTermination, TokenUsageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.model_context import BufferedChatCompletionContext

from autogen_core.memory import Memory

with open("./model.yaml", "r") as f:
    model_config = yaml.safe_load(f)

model_client = OpenAIChatCompletionClient.load_component(model_config)

model_client = OpenAIChatCompletionClient(
    model = 'gpt-3.5-turbo',
    api_key="-",
    base_url="https://api.openai-hk.com/v1/",
    max_retries = 5
)

prompt_list = ['Revise this with your best effort', 'Help me polish this', 'Rewrite this for me', 
                'Make this fluent while doing minimal change', 'Refine this for me please', 'Concise this for me and keep all the information',
                'Improve this in GPT way']

# prompt_list = ['Revise the code with your best effort', 'Help me polish this code', 'Rewrite the code with GPT style', 'Refine the code for me please', 'Concise the code without change the functionality']

prefix = "\nNo need to explain. Just write code without any explanation!: "
# 考虑使用用户交互智能体进行改进，比如应用与攻防交互或者进一步增强检测能力
# memory = Memory()

# 
rewrite_agent = AssistantAgent(
    "rewrite",
    model_client = model_client,
    system_message = "You are a helpful agent to do text rewriting, you need to get a text and use given prompt to rewrite it",
    model_context=BufferedChatCompletionContext(buffer_size=1),
    
)

feature_extract_agent = AssistantAgent(
    "feature_extract",
    model_client = model_client,
    tools = {get_stat},
    system_message = "You are a helpful agent to call function in your tools, you just need to call your tool with passing a given Index as parameter",
    model_context=BufferedChatCompletionContext(buffer_size=1),
)

analysis_agent = AssistantAgent(
    "analysis", 
    model_client = model_client,
    tools = {classifier},
    system_message = "you are a helpful agent to call a xgboost_classifier function to dispose text feature vectors.",
    model_context=BufferedChatCompletionContext(buffer_size=1),
)

async def main():
    print(len(feature_vectors))
    data_range = 600
    datas = load_data('dataset/News/news_combine.json')
    try:
        for idx, item in enumerate(datas[: data_range]):
            if (idx < len(feature_vectors)):
                continue
            
            rewrite_item = {
                "Index":item["Index"],
                "Text":item["Text"],
                "Source":item["Source"]
            }
            # rewrite_item = {
            #     "Index":item["Index"],  
            #     # "Text":("Explanation: " + item["Explanation"] + "\nImplementation: " + item["Implementation"]),
            #     "Text":("Implementation: " + item["Implementation"]),
            #     "Source":item["Source"]
            # }
            rewrite_item["Text"] = get_first_1024_tokens(rewrite_item["Text"])
            print(idx)
            print(rewrite_item["Text"])
            await rewrite_agent.on_reset(cancellation_token=CancellationToken())
            for prompt in prompt_list:
                await rewrite_agent.on_reset(cancellation_token=CancellationToken())
                # response = await rewrite_agent.on_messages(
                #     [TextMessage(content = f"{prompt} for the code Implementation part: \"{rewrite_item['Text']}\" \nNo need to explain. Just write code without any explanation!: ", source = "user")],
                #     cancellation_token=CancellationToken()
                # )
                response = await rewrite_agent.on_messages (
                    [TextMessage(content = f"{prompt}: \"{rewrite_item['Text']}\"", source = "user")],
                    cancellation_token=CancellationToken()
                )
                rewritten_text = response.chat_message.content
                print(rewritten_text)
                save_rewrite_data(rewritten_text, prompt, rewrite_item)
        
            rewrite_data.append(rewrite_item)
            print("rewriting has finished!")
        
            feature_extract_agent.run(task = f"pass {idx} to your tools and run it.")
            # get_stat(idx)
            if len(feature_vectors) == idx + 1:
                print("estimate call successed!")
            print(feature_vectors[idx])
            
        await analysis_agent.run(task = f"Call your function with this data_range parameter: {data_range}.")
        # classifier(data_range)
        save_rewrite(rewrite_data)
        save_features(feature_vectors)
    
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        save_rewrite(rewrite_data, "News_result/rewrite_data.json")
        save_features(feature_vectors, "News_result/feature_vectors.json")
        
    return 

asyncio.run(main())


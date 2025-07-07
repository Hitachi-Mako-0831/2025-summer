import asyncio
from autogen_core.models import CreateResult, UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient

model_client = OpenAIChatCompletionClient(
    model = 'gpt-4o',
    api_key="hk-p5ioqc10000518971d72d52ca5b76fcd3c56b2c5dac4a312",
    base_url="https://api.openai-hk.com/v1/",
    max_retries = 2
)

messages = [
    UserMessage(content="Write a very short story about a dragon.", source="user"),
]

# Create a stream.
stream = model_client.create_stream(messages=messages)

# Iterate over the stream and print the responses.
async def main():
    
    print("Streamed responses:")
    async for chunk in stream:  # type: ignore
        if isinstance(chunk, str):
            # The chunk is a string.
            print(chunk, flush=True, end="")
        else:
            # The final chunk is a CreateResult object.
            assert isinstance(chunk, CreateResult) and isinstance(chunk.content, str)
            # The last response is a CreateResult object with the complete message.
            print("\n\n------------\n")
            print("The complete response:", flush=True)
            print(chunk.content, flush=True)

if __name__ == "__main__":
    asyncio.run(main())
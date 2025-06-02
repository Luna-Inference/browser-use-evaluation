import asyncio
from dotenv import load_dotenv
load_dotenv()
from browser_use import Agent
from langchain_ollama import ChatOllama

async def main():
    # Using Gemma 3 4B model
    agent = Agent(
        task="Compare the price of gpt-4o and DeepSeek-V3",
        llm=ChatOllama(
            model="qwen3:1.7b",  # Gemma 3 4B parameter model
            num_ctx=8192,  # Context window size (Gemma 3 default)
            temperature=0.1,  # Lower temperature for more focused responses
        ),
       #  max_actions_per_step=3,
        # tool_call_in_content=False,  # Important for proper tool calling
    )
    
    # Capture and display the result
    result = await agent.run(max_steps=15)
    
    print("Gemma 3 4B Agent Result:")
    print("=" * 50)
    print(result)
    
    return result

if __name__ == "__main__":
    result = asyncio.run(main())
import asyncio
from dotenv import load_dotenv
load_dotenv()
from browser_use import Agent
from langchain_openai import ChatOpenAI

async def main():
    agent = Agent(
        task="Compare the price of gpt-4o and DeepSeek-V3",
        llm=ChatOpenAI(model="gpt-4o"),
    )
    
    # Capture the result
    result = await agent.run()
    
    # Display the result
    print("Agent Result:")
    print("=" * 50)
    print(result)
    
    # If you want to access specific parts of the result
    # (the exact structure depends on browser_use implementation)
    if hasattr(result, 'message'):
        print(f"\nFinal Message: {result.message}")
    
    if hasattr(result, 'success'):
        print(f"Success: {result.success}")
    
    return result

# Run and capture the result at the top level too
if __name__ == "__main__":
    result = asyncio.run(main())
import os
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

async def test_llm(name, base_url, api_key, model):
    print(f"\n--- Testing {name} ---")
    print(f"Base URL: {base_url}")
    print(f"Model: {model}")
    
    if not base_url or not model:
        print(f"[FAIL] {name} is missing base_url or model configuration.")
        return

    client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key
    )
    
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Directly provide the answer without any internal thought process or <think> tags."},
                {"role": "user", "content": "Testing. Please reply with 'Successfully connected'."}
            ],
            max_tokens=500,
            stream=False
        )
        # print(f"DEBUG: Complete response: {response}")
        
        if not response.choices:
            print(f"[FAIL] Response was empty (no choices). Full response: {response}")
            return
            
        reply = response.choices[0].message.content
        if reply:
            print(f"[OK] Success! Response: '{reply.strip()}'")
        else:
            print(f"[WARN] Success but empty content!")
            print(f"Finish Reason: {response.choices[0].finish_reason}")
            print(f"Message: {response.choices[0].message}")
    except Exception as e:
        print(f"[ERROR] Error connecting to {name}: {e}")
    finally:
        await client.close()

async def main():
    print("Testing LLM API Keys from .env...\n")
    
    # Test primary LLM
    await test_llm(
        name="Primary LLM (LLM_API_KEY)",
        base_url=os.getenv("LLM_BASE_URL"),
        api_key=os.getenv("LLM_API_KEY", ""),
        model=os.getenv("LLM_MODEL")
    )
    
    # Test Flash LLM
    await test_llm(
        name="Flash LLM (LLM_FLASH_API_KEY)",
        base_url=os.getenv("LLM_FLASH_BASE_URL"),
        api_key=os.getenv("LLM_FLASH_API_KEY", ""),
        model=os.getenv("LLM_FLASH_MODEL")
    )

if __name__ == "__main__":
    asyncio.run(main())

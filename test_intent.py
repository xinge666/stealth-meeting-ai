import asyncio
import sys
import logging
from src.config import AppConfig
from src.event_bus import EventBus
from src.intelligence.llm_client import LLMClient

# Setup logging & fix Windows output encoding
sys.stdout.reconfigure(encoding='utf-8')
logging.basicConfig(level=logging.INFO, format="%(message)s")

async def test_intent_analysis():
    # Load config and override LLM config to use flash_llm specifically for intent analysis
    config = AppConfig.from_env()
    
    # In test, we explicitly use flash_llm parameters locally bound to an LLMClient
    print(f"Testing Intent Recognition with Model: {config.flash_llm.model}")
    print(f"Base URL: {config.flash_llm.base_url}")
    
    bus = EventBus()
    # Explicitly pass the flash_llm config block to test intent analysis feature
    client = LLMClient(config=config.flash_llm, event_bus=bus)
    await client.initialize()
    
    # Define test cases for ASR output
    test_cases = [
        # Substantive questions
        {"text": "那个，能不能给我介绍一下 React 的生命周期是怎么样的？", "history": ""},
        {"text": "哎，就是那个，transformer 为什么要除以根号 dk 啊？", "history": "面试官：请问你对 transformer 了解多少？"},
        
        # Follow-up/Pronoun substitutions
        {"text": "那它的时间复杂度是多少呀？", "history": "面试官刚刚问到了快速排序算法。"},
        {"text": "刚才说的那个机制，在多线程下安全吗？", "history": "关于 Java 中的 HashMap，你有什么看法？"},
        
        # Idle chat/Noise
        {"text": "嗯，对对，是的，好的。", "history": "好的，我们接下来看下一个问题。"},
        {"text": "那个...呃...这个...我想想啊", "history": ""},
        
        # Boundary cases (noise + question)
        {"text": "好的老师，我的问题是，呃，就是实习期间能不能转正呢？", "history": ""},
        {"text": "大家下午好！我是今天的候选人。请问今天的面试流程是怎样的呀？", "history": ""},
    ]
    
    print("\nStarting Intent Analysis Tests...\n")
    
    for i, case in enumerate(test_cases, 1):
        print("=" * 60)
        print(f"Test Case {i}:")
        print(f"[ASR Text] {case['text']}")
        if case['history']:
            print(f"[History]  {case['history']}")
            
        print("-" * 30)
        result = await client.analyze_intent(case["text"], case["history"])
        
        print(f"Is Question:        {result.get('is_question')}")
        print(f"Confidence:         {result.get('confidence')}")
        print(f"Extracted Question: {result.get('extracted_question')}")
        print("=" * 60 + "\n")
        
    await client.close()

if __name__ == "__main__":
    asyncio.run(test_intent_analysis())

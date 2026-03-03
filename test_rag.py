import asyncio
import os
import sys
from src.intelligence.rag import RAGEngine

# Fix Windows console print encoding issues
sys.stdout.reconfigure(encoding='utf-8')

async def test_rag():
    docs_dir = os.path.join(os.path.dirname(__file__), "data", "docs")
    engine = RAGEngine(docs_dir=docs_dir)
    await engine.initialize()
    
    print(f"Total slices indexed: {len(engine.slices)}")
    print(f"Total blocks indexed: {len(engine.blocks)}")
    
    queries = [
        "React 的生命周期是什么？",
        "如何准备系统设计面试",
        "transformer 为什么要除以 dk"
    ]
    
    for q in queries:
        print(f"\n======================================")
        print(f"--- Query: {q} ---")
        results = await engine.search(q, top_k=2)
        for i, res in enumerate(results):
            print(f"\nResult {i+1} [Score: {res['score']:.4f}]")
            print(f"Source: {res['source']} (Matched Slice: {res['matched_slice_id']})")
            print(f"Matched Text (Snippet): {res['matched_slice_text'][:100]}...")
            print("--- Full Knowledge Block Context ---")
            print(res['text'])
            print("---------------------------------")

if __name__ == "__main__":
    asyncio.run(test_rag())

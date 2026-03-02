import asyncio
import os
from src.intelligence.rag import RAGEngine

async def test_rag():
    docs_dir = os.path.join(os.path.dirname(__file__), "data", "docs")
    engine = RAGEngine(docs_dir=docs_dir)
    await engine.initialize()
    
    print(f"Total chunks indexed: {len(engine.global_chunks)}")
    
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
            print(f"Source: {res['source']} (Matched Chunk Index: {res['matched_chunk_idx']})")
            print("--- Context Window (-3 to +3) ---")
            print(res['text'])
            print("---------------------------------")

if __name__ == "__main__":
    asyncio.run(test_rag())

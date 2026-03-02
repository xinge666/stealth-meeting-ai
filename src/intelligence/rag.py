"""
Lightweight RAG Engine using Sentence Transformers embeddings.
Provides chunks of 64 tokens and context window of 3 chunks.
"""

import os
import logging
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class RAGChunk:
    def __init__(self, text: str, source: str, doc_chunk_idx: int):
        self.text = text
        self.source = source
        self.doc_chunk_idx = doc_chunk_idx

class DocumentChunks:
    def __init__(self, source: str):
        self.source = source
        self.chunks = [] # list of text chunks

class RAGEngine:
    def __init__(self, docs_dir: str):
        self.docs_dir = docs_dir
        self.global_chunks: list[RAGChunk] = []
        # Fast lookup for document context
        self.doc_registry: dict[str, list[str]] = {}
        self.embeddings = None
        
        # Load m3e model for embedding
        logger.info("Loading SentenceTransformer model 'moka-ai/m3e-base'...")
        self.model = SentenceTransformer('moka-ai/m3e-base')

    async def initialize(self):
        """Loads documents from docs_dir, chunks them, and builds embeddings."""
        logger.info("Initializing RAG Engine from: %s", self.docs_dir)
        
        if not os.path.exists(self.docs_dir):
            logger.warning("Docs directory %s does not exist. RAG will be empty.", self.docs_dir)
            return
            
        # Optional: you could make this async, but huggingface tokenizer and encode are sync

        for root, _, files in os.walk(self.docs_dir):
            for file in files:
                if file.endswith('.md') or file.endswith('.txt'):
                    path = os.path.join(root, file)
                    rel_source = os.path.relpath(path, self.docs_dir)
                    try:
                        with open(path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            
                        # Chunking by 64 tokens using the model's tokenizer
                        tokens = self.model.tokenizer(content, add_special_tokens=False)['input_ids']
                        chunk_size = 64
                        
                        doc_chunks = []
                        for i in range(0, len(tokens), chunk_size):
                            chunk_tokens = tokens[i:i + chunk_size]
                            chunk_text = self.model.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                            doc_chunks.append(chunk_text)
                            
                        self.doc_registry[rel_source] = doc_chunks
                        
                        # Add to global chunks for index
                        for idx, text in enumerate(doc_chunks):
                            self.global_chunks.append(RAGChunk(text, rel_source, idx))
                            
                    except Exception as e:
                        logger.error("Failed to read %s: %s", path, e)
                        
        logger.info("Loaded %d chunks. Building embeddings index...", len(self.global_chunks))
        if self.global_chunks:
            texts = [c.text for c in self.global_chunks]
            # convert to tensor to enable fast metrics
            self.embeddings = self.model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
        logger.info("RAG Index built successfully.")

    async def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Searches the index and returns context window of +/- 3 chunks."""
        if not self.global_chunks or self.embeddings is None:
            return []

        query_emb = self.model.encode([query], convert_to_tensor=True, normalize_embeddings=True)
        cos_scores = torch.nn.functional.cosine_similarity(query_emb, self.embeddings).cpu()
        
        top_k = min(top_k, len(self.global_chunks))
        top_results = torch.topk(cos_scores, k=top_k)
        
        results = []
        for score, idx in zip(top_results[0].tolist(), top_results[1].tolist()):
            chunk = self.global_chunks[idx]
            doc_chunks = self.doc_registry[chunk.source]
            
            # Context window +/- 3
            start_idx = max(0, chunk.doc_chunk_idx - 3)
            end_idx = min(len(doc_chunks), chunk.doc_chunk_idx + 4) # +4 because slice is exclusive
            
            context_pieces = []
            for i in range(start_idx, end_idx):
                prefix = ">> " if i == chunk.doc_chunk_idx else "   "
                context_pieces.append(f"{prefix}[Chunk {i}] {doc_chunks[i]}")
                
            context_text = "\n".join(context_pieces)
            
            results.append({
                "score": score,
                "text": context_text, # this is the context window content
                "source": chunk.source,
                "matched_chunk_idx": chunk.doc_chunk_idx,
                "matched_text": chunk.text
            })
            
        return results

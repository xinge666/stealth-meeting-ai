"""
Lightweight RAG Engine using Sentence Transformers embeddings.
Provides structured chunking and embedding logic.
"""

import os
import logging
import torch
import xxhash
import pickle
import re
from typing import List, Dict, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

@dataclass
class KnowledgeBlock:
    """A complete knowledge block, like a section in Markdown or a full paragraph."""
    block_id: str
    source: str
    title: str
    content: str
    
@dataclass
class TextSlice:
    """A sliced piece of text used for embeddding and semantic search."""
    slice_id: str
    block_id: str
    text: str
    source: str


class BaseChunker:
    def chunk(self, content: str, source: str) -> List[KnowledgeBlock]:
        raise NotImplementedError

class MarkdownChunker(BaseChunker):
    """Chunks markdown by headings to create coherent blocks of knowledge."""
    def chunk(self, content: str, source: str) -> List[KnowledgeBlock]:
        blocks = []
        # Split by heading taking into account AT LEAST 1 hash and a space.
        # This regex isolates the header line.
        lines = content.split('\n')
        
        current_title = "Document Start"
        current_content = []
        
        block_idx = 0
        for line in lines:
            if re.match(r'^#{1,6}\s+', line):
                # Save previous block if not empty
                text_content = '\n'.join(current_content).strip()
                if text_content:
                    block_id = f"{source}#block_{block_idx}"
                    blocks.append(KnowledgeBlock(
                        block_id=block_id,
                        source=source,
                        title=current_title,
                        content=text_content
                    ))
                    block_idx += 1
                
                # Start new block
                current_title = line.strip()
                current_content = [line]
            else:
                current_content.append(line)
                
        # Save last block
        text_content = '\n'.join(current_content).strip()
        if text_content:
            block_id = f"{source}#block_{block_idx}"
            blocks.append(KnowledgeBlock(
                block_id=block_id,
                source=source,
                title=current_title,
                content=text_content
            ))
            
        return blocks

class FixedLengthChunker(BaseChunker):
    """Fallback chunker for plain text that chunks by fixed length."""
    def __init__(self, block_size: int = 1000):
        self.block_size = block_size
        
    def chunk(self, content: str, source: str) -> List[KnowledgeBlock]:
        blocks = []
        text_length = len(content)
        block_idx = 0
        
        for i in range(0, text_length, self.block_size):
            chunk_text = content[i:i + self.block_size].strip()
            if chunk_text:
                block_id = f"{source}#block_{block_idx}"
                blocks.append(KnowledgeBlock(
                    block_id=block_id,
                    source=source,
                    title=f"Chunk {block_idx}",
                    content=chunk_text
                ))
                block_idx += 1
                
        return blocks

class SlidingWindowSlicer:
    """Slices a text block into sliding windows for embedding."""
    def __init__(self, slice_size: int = 300, overlap: int = 50):
        self.slice_size = slice_size
        self.overlap = overlap
        
    def slice(self, block: KnowledgeBlock) -> List[TextSlice]:
        slices = []
        text = block.content
        text_length = len(text)
        
        step = self.slice_size - self.overlap
        if step <= 0:
            step = self.slice_size # fallback if overlap is too big
            
        slice_idx = 0
        for i in range(0, text_length, step):
            slice_text = text[i:i + self.slice_size].strip()
            if slice_text:
                slices.append(TextSlice(
                    slice_id=f"{block.block_id}_slice_{slice_idx}",
                    block_id=block.block_id,
                    text=slice_text,
                    source=block.source
                ))
                slice_idx += 1
                
        return slices

class RAGEngine:
    def __init__(self, docs_dir: str):
        self.docs_dir = docs_dir
        self.model_name = 'moka-ai/m3e-base'
        
        self.blocks: Dict[str, KnowledgeBlock] = {}
        self.slices: List[TextSlice] = []
        self.embeddings: Optional[torch.Tensor] = None
        
        self.cache_dir = os.path.join(os.path.dirname(__file__), ".rag_cache", self.model_name.replace("/", "_"))
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load m3e model for embedding
        logger.info(f"Loading SentenceTransformer model '{self.model_name}'...")
        self.model = SentenceTransformer(self.model_name)
        
        self.markdown_chunker = MarkdownChunker()
        self.fixed_chunker = FixedLengthChunker()
        self.slicer = SlidingWindowSlicer(slice_size=300, overlap=50)

    def _get_file_hash(self, content: str) -> str:
        return xxhash.xxh64(content.encode('utf-8')).hexdigest()

    async def initialize(self):
        """Loads documents from docs_dir, chunks them, and builds embeddings with caching."""
        logger.info("Initializing RAG Engine from: %s", self.docs_dir)
        
        if not os.path.exists(self.docs_dir):
            logger.warning("Docs directory %s does not exist. RAG will be empty.", self.docs_dir)
            return

        all_slices = []
        all_blocks = {}
        
        # We will collect slices that need to be embedded in this run
        slices_to_embed = []
        new_embeddings_list = []
        
        cached_slices_list = []
        cached_embeddings_list = []

        for root, _, files in os.walk(self.docs_dir):
            for file in files:
                if file.endswith('.md') or file.endswith('.txt'):
                    path = os.path.join(root, file)
                    rel_source = os.path.relpath(path, self.docs_dir)
                    try:
                        with open(path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            
                        file_hash = self._get_file_hash(content)
                        cache_path = os.path.join(self.cache_dir, f"{file_hash}.pkl")
                        
                        if os.path.exists(cache_path):
                            logger.info(f"Loading cached embeddings for {rel_source}")
                            with open(cache_path, 'rb') as f:
                                cache_data = pickle.load(f)
                                
                            file_blocks = cache_data['blocks']
                            file_slices = cache_data['slices']
                            file_embeddings = cache_data['embeddings']
                            
                            all_blocks.update(file_blocks)
                            cached_slices_list.extend(file_slices)
                            cached_embeddings_list.append(file_embeddings)
                            
                        else:
                            logger.info(f"Processing and chunking {rel_source}")
                            if file.endswith('.md'):
                                blocks = self.markdown_chunker.chunk(content, rel_source)
                            else:
                                blocks = self.fixed_chunker.chunk(content, rel_source)
                                
                            file_blocks = {b.block_id: b for b in blocks}
                            all_blocks.update(file_blocks)
                            
                            file_slices = []
                            for block in blocks:
                                file_slices.extend(self.slicer.slice(block))
                                
                            if file_slices:
                                # Compute embeddings for this file
                                texts = [s.text for s in file_slices]
                                file_embeddings = self.model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
                                
                                # Save to cache
                                cache_data = {
                                    'blocks': file_blocks,
                                    'slices': file_slices,
                                    'embeddings': file_embeddings
                                }
                                with open(cache_path, 'wb') as f:
                                    pickle.dump(cache_data, f)
                                    
                                slices_to_embed.extend(file_slices)
                                new_embeddings_list.append(file_embeddings)
                                
                    except Exception as e:
                        logger.error("Failed to read or process %s: %s", path, e)
                        
        self.blocks = all_blocks
        self.slices = cached_slices_list + slices_to_embed
        
        all_embeddings = cached_embeddings_list + new_embeddings_list
        if all_embeddings:
            self.embeddings = torch.cat(all_embeddings, dim=0)
            logger.info("Loaded %d slices in total. RAG Index built successfully.", len(self.slices))
        else:
            self.embeddings = None
            logger.info("No documents found or embedded.")


    async def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Searches the index and returns the full KnowledgeBlock logic."""
        if not self.slices or self.embeddings is None:
            return []

        query_emb = self.model.encode([query], convert_to_tensor=True, normalize_embeddings=True)
        cos_scores = torch.nn.functional.cosine_similarity(query_emb, self.embeddings).cpu()
        
        top_k = min(top_k, len(self.slices))
        top_results = torch.topk(cos_scores, k=top_k)
        
        results = []
        seen_blocks = set()
        
        for score, idx in zip(top_results[0].tolist(), top_results[1].tolist()):
            slice_obj = self.slices[idx]
            block_id = slice_obj.block_id
            
            # We want to return unique blocks for the user context
            if block_id in seen_blocks:
                continue
                
            seen_blocks.add(block_id)
            block = self.blocks.get(block_id)
            if not block:
                continue
            
            # Formating block string output: include header/title and full content.
            block_text = f"[{block.title}]\n{block.content}"
            
            results.append({
                "score": score,
                "text": block_text, 
                "source": block.source,
                "matched_slice_id": slice_obj.slice_id,
                "matched_slice_text": slice_obj.text
            })
            
            # If we collected enough unique blocks, we can break
            # if len(results) >= top_k:
            #     break
            
        return results

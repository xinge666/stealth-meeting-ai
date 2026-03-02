# AI Agent 面试题库 - 编程实战篇

## 📚 适用对象
- ✅ 算法工程师（手撕核心算法）
- ✅ 开发工程师（实现系统模块）
- ⏱️ 建议学习时间：3-5天

## 第一部分：LLM 基础编程（必备）

**Q1: 手撕 Self-Attention 机制**
- 难度：⭐⭐⭐
- 时间：30分钟
- 语言：Python + PyTorch
- 标签：#Attention #Transformer #手撕代码
- 公司：字节、阿里、腾讯（高频）

**要求**：实现自注意力机制，输入 Q, K, V (batch, seq_len, d_model)，输出 attention output

**【答案参考】**
```python
import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        # x.shape: (batch_size, seq_len, d_model)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # scores: (batch_size, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_model)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attn_weights = torch.softmax(scores, dim=-1)
        # output: (batch_size, seq_len, d_model)
        output = torch.matmul(attn_weights, V)
        return output, attn_weights
```

---

**Q2: 实现 BPE Tokenizer**
- 难度：⭐⭐⭐
- 时间：40分钟
- 语言：Python
- 标签：#Tokenizer #BPE #编码

**要求**：从零实现 BPE 分词算法，包括训练和编码两个阶段

**【答案参考】**
```python
import collections
import re

class BPETokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = set()

    def get_stats(self, vocab):
        pairs = collections.defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                pairs[symbols[i], symbols[i+1]] += freq
        return pairs
        
    def merge_vocab(self, pair, v_in):
        v_out = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in v_in:
            w_out = p.sub(''.join(pair), word)
            v_out[w_out] = v_in[word]
        return v_out

    def train(self, texts):
        # 初始化带空格的字符频次字典
        words = ' '.join(texts).split()
        word_freq = collections.Counter(words)
        vocab = {' '.join(k) + ' </w>': v for k, v in word_freq.items()}
        
        num_merges = self.vocab_size - len(set("".join(words)))
        for i in range(num_merges):
            pairs = self.get_stats(vocab)
            if not pairs: break
            best = max(pairs, key=pairs.get)
            vocab = self.merge_vocab(best, vocab)
            self.merges[best] = ''.join(best)
            
    def encode(self, text):
        words = text.split()
        encoded = []
        for word in words:
            word = ' '.join(word) + ' </w>'
            while True:
                pairs = collections.Counter()
                symbols = word.split()
                for i in range(len(symbols)-1):
                    pair = (symbols[i], symbols[i+1])
                    if pair in self.merges:
                        pairs[pair] = 1 # found a possible merge
                if not pairs: break
                # merge the first matching pair we learned
                best_pair = next(iter(pairs.keys())) 
                word = word.replace(' '.join(best_pair), self.merges[best_pair])
            encoded.extend(word.split())
        return encoded
```

---

**Q3: 实现 Top-K/Top-P 采样**
- 难度：⭐⭐
- 时间：20分钟
- 语言：Python + PyTorch
- 标签：#采样 #解码策略

**要求**：实现 Top-K 和 Nucleus (Top-P) 采样算法

**【答案参考】**
```python
import torch

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """
    logits: (batch_size, vocab_size)
    """
    if top_k > 0:
        # 移除除了top_k之外的所有token
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # 移除累积概率超过top_p的token (但保持至少一个token)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        for i in range(logits.shape[0]):
            indices = sorted_indices[i][sorted_indices_to_remove[i]]
            logits[i][indices] = filter_value
            
    return logits
    
# 使用示例:
# logits = model(input_ids)
# filtered_logits = top_k_top_p_filtering(logits[:, -1, :], top_k=50, top_p=0.9)
# final_probs = torch.softmax(filtered_logits, dim=-1)
# next_token = torch.multinomial(final_probs, num_samples=1)
```

---

**Q4: 实现 Multi-Head Attention**
- 难度：⭐⭐⭐
- 时间：35分钟
- 语言：Python + PyTorch
- 标签：#MHA #Transformer

**要求**：实现多头注意力机制，支持可配置的头数

**【答案参考】**
```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # 1. 线性投影并分头: (batch, seq_len, num_heads, d_k) -> (batch, num_heads, seq_len, d_k)
        Q = self.W_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. 计算注意力并应用 mask
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            # mask shape 需 broadcastable，比如 (batch, 1, seq_len, seq_len)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(scores, dim=-1)
        
        # 3. 得到 multi-head 输出：(batch, num_heads, seq_len, d_k)
        context = torch.matmul(attn_weights, V)
        
        # 4. 拼接并通过终投影: (batch, seq_len, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)
        return output
```

---

**Q5: 实现 ROPE 位置编码**
- 难度：⭐⭐⭐⭐
- 时间：45分钟
- 语言：Python + PyTorch
- 标签：#ROPE #位置编码

**要求**：实现旋转位置编码（Rotary Position Embedding）

**【答案参考】**见第七部分 Q30 具体解答。

---

## 第二部分：Agent 核心模块

**Q6: 手撕 ReAct Agent**
- 难度：⭐⭐⭐⭐
- 时间：45分钟
- 语言：Python
- 标签：#ReAct #Agent #框架
- 公司：字节、阿里（高频）

**要求**：实现 ReAct 框架，支持 Thought → Action → Observation 循环

**【答案参考】**
```python
import re

class ReActAgent:
    def __init__(self, llm_fn, tools):
        self.llm = llm_fn
        self.tools = {tool.name: tool.func for tool in tools}
        self.system_prompt = """You are a helpful assistant. Solve the task using the following exact format:
Thought: think about what to do next
Action: the action to take (must be one of {tool_names})
Action Input: the input for the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I know the final answer
Final Answer: the final answer"""
        self.tool_names = ", ".join(self.tools.keys())
        
    def run(self, user_query, max_steps=5):
        prompt = self.system_prompt.format(tool_names=self.tool_names) + f"\n\nTask: {user_query}\n"
        
        for _ in range(max_steps):
            response = self.llm(prompt)
            print(f"LLM: \n{response}\n")
            prompt += response + "\n"
            
            # 检查是否达成最终答案
            if "Final Answer:" in response:
                return response.split("Final Answer:")[1].strip()
                
            # 解析 Action 和 Action Input
            action_match = re.search(r"Action:\s*(.*)", response)
            input_match = re.search(r"Action Input:\s*(.*)", response)
            
            if action_match and input_match:
                action = action_match.group(1).strip()
                action_input = input_match.group(1).strip()
                
                if action in self.tools:
                    try:
                        observation = str(self.tools[action](action_input))
                    except Exception as e:
                        observation = f"Error executing {action}: {e}"
                else:
                    observation = f"Invalid action: {action}"
                    
                print(f"Observation: {observation}\n")
                prompt += f"Observation: {observation}\n"
            else:
                prompt += "Observation: Format error. Please provide Action and Action Input.\n"
                
        return "Failed to reach final answer."
```

---

**Q7: 实现 Tool Registry 工具注册系统**
- 难度：⭐⭐⭐
- 时间：30分钟
- 语言：Python
- 标签：#ToolUse #Agent

**要求**：实现工具注册、查询、调用的完整系统

**【答案参考】**
```python
import inspect

class Tool:
    def __init__(self, name, description, func, schema):
        self.name = name
        self.description = description
        self.func = func
        self.schema = schema

class ToolRegistry:
    def __init__(self):
        self.registry = {}
        
    def register(self, name, description):
        def decorator(func):
            # 获取函数的签名作为 schema (简单示例)
            sig = inspect.signature(func)
            schema = {p_name: str(p.annotation) for p_name, p in sig.parameters.items()}
            self.registry[name] = Tool(name, description, func, schema)
            return func
        return decorator
        
    def get_tool(self, name):
        return self.registry.get(name)
        
    def list_tools(self):
        return [{"name": tool.name, "description": tool.description, "schema": tool.schema} 
                for tool in self.registry.values()]
                
    def call(self, name, **kwargs):
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool {name} not found")
        return tool.func(**kwargs)

# 示例:
# registry = ToolRegistry()
# @registry.register("calculator", "Evaluate math expression")
# def calc(exp: str): return eval(exp)
```

---

**Q8: 实现 Memory 系统（短期+长期记忆）**
- 难度：⭐⭐⭐⭐
- 时间：50分钟
- 语言：Python
- 标签：#Memory #Agent #存储

**要求**：实现对话历史管理（短期）+ 向量检索（长期记忆）

**【答案参考】**
```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class HybridMemory:
    def __init__(self, max_short_term=5, embed_fn=None):
        self.max_short_term = max_short_term
        self.short_term = [] # [(role, msg), ...]
        self.long_term_texts = []
        self.long_term_vectors = []
        self.embed_fn = embed_fn or (lambda x: np.random.rand(768)) # 默认占位符
        
    def add_message(self, role, msg):
        # 存入短期记忆
        self.short_term.append((role, msg))
        if len(self.short_term) > self.max_short_term * 2:
            # 短期记忆溢出，存入长期记忆，并清理短期记忆
            evicted_text = f"{self.short_term[0][0]}: {self.short_term[0][1]}"
            self.long_term_texts.append(evicted_text)
            self.long_term_vectors.append(self.embed_fn(evicted_text))
            self.short_term.pop(0)

    def retrieve_long_term(self, query, top_k=2):
        if not self.long_term_vectors: return []
        q_vec = self.embed_fn(query).reshape(1, -1)
        vectors = np.vstack(self.long_term_vectors)
        sim_scores = cosine_similarity(q_vec, vectors)[0]
        top_indices = np.argsort(sim_scores)[-top_k:][::-1]
        return [self.long_term_texts[i] for i in top_indices]
        
    def get_context(self, query):
        past_memories = self.retrieve_long_term(query)
        context = "Relevant Past Memories:\n" + "\n".join(past_memories) + "\n\n"
        context += "Recent Chat History:\n"
        for role, msg in self.short_term:
            context += f"{role}: {msg}\n"
        return context
```

---

**Q9: 实现 Chain-of-Thought Prompting**
- 难度：⭐⭐
- 时间：20分钟
- 语言：Python
- 标签：#CoT #Prompt #推理

**【答案参考】**
```python
def generate_cot_prompt(question, examples=None):
    prompt = "Please solve the problem step-by-step.\n\n"
    if examples:
        for ex in examples:
            prompt += f"Q: {ex['q']}\n"
            prompt += f"A: Let's think step by step.\n{ex['steps']}\nTherefore, the answer is {ex['ans']}.\n\n"
            
    prompt += f"Q: {question}\n"
    prompt += "A: Let's think step by step.\n"
    return prompt
```

---

**Q10: 实现 Self-Reflection 自我反思机制**
- 难度：⭐⭐⭐⭐
- 时间：40分钟
- 语言：Python
- 标签：#Reflection #Agent #优化

**【答案参考】**
```python
class ReflectiveAgent:
    def __init__(self, generator_llm, critic_llm):
        self.gen_llm = generator_llm
        self.critic_llm = critic_llm
        
    def solve(self, task, max_attempts=3):
        draft = self.gen_llm(f"Please solve: {task}")
        
        for attempt in range(max_attempts):
            critic_prompt = f"Task: {task}\nDraft Answer: {draft}\n"
            critic_prompt += "Evaluate this answer. If perfect, reply 'PASS'. Else, suggest specific improvements."
            feedback = self.critic_llm(critic_prompt)
            
            if "PASS" in feedback:
                return draft
                
            refine_prompt = f"Task: {task}\nPrevious Draft: {draft}\nCritic Feedback: {feedback}\n"
            refine_prompt += "Please provide an improved answer incorporating the feedback."
            draft = self.gen_llm(refine_prompt)
            
        return draft # 返回最终修改后的答案
```

---

## 第三部分：RAG 系统实现

**Q11: 实现文档切块策略（Chunking）**
- 难度：⭐⭐
- 时间：20分钟
- 语言：Python
- 标签：#RAG #文档处理 #切块
- 公司：阿里、腾讯（常考）

**要求**：实现固定大小切块、重叠切块、语义切块

**【答案参考】**
```python
import re

def fixed_size_chunking(text, chunk_size=100, overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks

def semantic_chunking(text, max_len=200):
    # 简单的按标点符号进行语义切块
    sentences = re.split(r'(?<=[。！？])\s*', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_len:
            current_chunk += sentence
        else:
            if current_chunk: chunks.append(current_chunk)
            current_chunk = sentence
    if current_chunk: chunks.append(current_chunk)
    return chunks
```

---

**Q12: 实现混合检索（BM25 + 向量检索）**
- 难度：⭐⭐⭐
- 时间：35分钟
- 语言：Python
- 标签：#RAG #混合检索 #BM25

**【答案参考】**
```python
import numpy as np

def rrf_score(rank, k=60):
    return 1 / (k + rank)

class HybridRetriever:
    def __init__(self, bm25_retriever, vector_retriever):
        self.bm25_retriever = bm25_retriever
        self.vector_retriever = vector_retriever
        
    def retrieve(self, query, top_k=5):
        # [(doc_id, score), ...]
        bm25_results = self.bm25_retriever.search(query) 
        vec_results = self.vector_retriever.search(query)
        
        # 赋予 rank 排名
        bm25_ranks = {doc_id: rank+1 for rank, (doc_id, _) in enumerate(bm25_results)}
        vec_ranks = {doc_id: rank+1 for rank, (doc_id, _) in enumerate(vec_results)}
        
        all_docs = set(bm25_ranks.keys()).union(set(vec_ranks.keys()))
        hybrid_scores = {}
        for doc in all_docs:
            score = 0
            if doc in bm25_ranks: score += rrf_score(bm25_ranks[doc])
            if doc in vec_ranks:  score += rrf_score(vec_ranks[doc])
            hybrid_scores[doc] = score
            
        sorted_docs = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:top_k]
```

---

**Q13: 实现 Reranker 重排序模块**
- 难度：⭐⭐⭐
- 时间：30分钟
- 语言：Python
- 标签：#RAG #Reranker #排序

**【答案参考】**
```python
import torch

class CrossEncoderReranker:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def rerank(self, query, documents, top_n=3):
        pairs = [[query, doc] for doc in documents]
        
        # tokenizer 处理对：[CLS] query [SEP] doc [SEP]
        inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        
        with torch.no_grad():
            scores = self.model(**inputs).logits.squeeze(-1) # 获取分类头或打分输出
            
        # 根据 score 降序排名
        scored_docs = list(zip(scores.tolist(), documents))
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        return [doc for score, doc in scored_docs[:top_n]]
```

---

**Q14: 实现 Semantic Cache 语义缓存**
- 难度：⭐⭐⭐
- 时间：30分钟
- 语言：Python
- 标签：#RAG #缓存 #优化

**【答案参考】**
```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SemanticCache:
    def __init__(self, embed_fn, threshold=0.9):
        self.embed_fn = embed_fn
        self.threshold = threshold
        self.queries = []
        self.q_embeddings = []
        self.answers = []
        
    def get(self, query):
        if not self.q_embeddings: return None
        
        q_vec = self.embed_fn(query).reshape(1, -1)
        matrix = np.vstack(self.q_embeddings)
        sim_scores = cosine_similarity(q_vec, matrix)[0]
        
        best_idx = np.argmax(sim_scores)
        if sim_scores[best_idx] >= self.threshold:
            print(f"Cache Hit! Score: {sim_scores[best_idx]}")
            return self.answers[best_idx]
        return None
        
    def set(self, query, answer):
        q_vec = self.embed_fn(query)
        self.queries.append(query)
        self.q_embeddings.append(q_vec)
        self.answers.append(answer)
```

---

**Q15: 实现 HyDE（假设性文档嵌入）**
- 难度：⭐⭐⭐⭐
- 时间：35分钟
- 语言：Python
- 标签：#RAG #HyDE #查询优化

**【答案参考】**
```python
class HyDERetriever:
    def __init__(self, llm_fn, embed_fn, vector_db):
        self.llm = llm_fn
        self.embed = embed_fn
        self.vector_db = vector_db
        
    def search(self, query, top_k=5):
        # 步骤1: 用 LLM 生成假设性段落
        prompt = f"Please write a passage to answer the question: {query}"
        hypothetical_doc = self.llm(prompt)
        
        # 步骤2: 将虚构文档转化为向量
        vector = self.embed(hypothetical_doc)
        
        # 步骤3: 去向量数据库中进行实际相似度召回
        results = self.vector_db.search_vector(vector, top_k=top_k)
        return results
```

---

## 第四部分：模型优化与推理

**Q16: 实现 KV Cache**
- 难度：⭐⭐⭐⭐
- 时间：45分钟
- 语言：Python + PyTorch
- 标签：#推理优化 #KVCache

**【答案参考】**
```python
import torch
import torch.nn as nn

class AttentionWithKVCache(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.d_model = d_model
        
    def forward(self, x, kv_cache=None):
        # x shape: (batch_size, 1, d_model) 用于逐步生成时seq_len=1
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1) # 各自持 (batch, 1, d_model)
        
        if kv_cache is not None:
            # 拼接到过去状态: kv_cache = (past_k, past_v)
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=1) # (batch, past_seq + 1, d_model)
            v = torch.cat([past_v, v], dim=1)
            
        new_kv_cache = (k, v)
        
        # 计算 Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_model ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        return self.out_proj(out), new_kv_cache
```

---

**Q17: 实现 Beam Search 解码**
- 难度：⭐⭐⭐
- 时间：35分钟
- 语言：Python
- 标签：#解码 #BeamSearch

**【答案参考】**
```python
import torch

def beam_search(model, start_token, beam_width, max_len, eos_token):
    # beam 包含 (sequence_list, log_prob)
    beams = [([start_token], 0.0)]
    
    for _ in range(max_len):
        all_candidates = []
        for seq, score in beams:
            # 如果序列已结束则不再扩展
            if seq[-1] == eos_token:
                all_candidates.append((seq, score))
                continue
                
            input_tensor = torch.tensor([seq])
            with torch.no_grad():
                logits = model(input_tensor)[0, -1, :] # 获取最后一个 token
            
            log_probs = torch.log_softmax(logits, dim=-1)
            # 取 top K 概率的 token
            topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)
            
            for i in range(beam_width):
                candidate_seq = seq + [topk_indices[i].item()]
                candidate_score = score + topk_log_probs[i].item()
                all_candidates.append((candidate_seq, candidate_score))
                
        # 基于分数对候选序列排序并截断回 beam_width 长度 (长度惩罚略)
        ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
        beams = ordered[:beam_width]
        
    return beams[0][0] # 返回得分最高的一整局
```

---

**Q18: 实现 LoRA 微调**
- 难度：⭐⭐⭐⭐⭐
- 时间：60分钟
- 语言：Python + PyTorch
- 标签：#LoRA #微调 #PEFT

**要求**：实现 LoRA 低秩适配器

**【答案参考】**
```python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, linear_layer: nn.Linear, r: int, alpha: int = 1, dropout: float = 0.0):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.linear = linear_layer
        self.r = r
        self.scaling = alpha / r
        
        # 冻结原始全连接层
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
            
        # 低秩权重矩阵 A (in -> r) 和 B (r -> out)
        self.lora_A = nn.Parameter(torch.zeros(self.in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, self.out_features))
        self.dropout = nn.Dropout(p=dropout)
        
        nn.init.normal_(self.lora_A, mean=0.0, std=1 / self.in_features)
        nn.init.zeros_(self.lora_B) # B 初始化为0保证初始对前向无影响
        
    def forward(self, x):
        # 原始线性前向
        base_out = self.linear(x)
        # 添加 LoRA 旁路: x * A * B * scale
        lora_out = torch.matmul(torch.matmul(self.dropout(x), self.lora_A), self.lora_B) * self.scaling
        return base_out + lora_out
```

---

## 第五部分：评估与监控

**Q19: 实现 BLEU/ROUGE 评估指标**
- 难度：⭐⭐
- 时间：25分钟
- 语言：Python
- 标签：#评估 #指标

**【答案参考】**
实现精确的 n-gram 重叠度计分（以单字/单个 token n=1 为例）：
```python
from collections import Counter
import math

def calculate_bleu_1(candidate, reference):
    cand_tokens = candidate.split()
    ref_tokens = reference.split()
    cand_counter = Counter(cand_tokens)
    ref_counter = Counter(ref_tokens)
    
    matches = 0
    for token in cand_counter:
        matches += min(cand_counter[token], ref_counter.get(token, 0))
    precision = matches / len(cand_tokens) if cand_tokens else 0
    
    # 短句惩罚项 (Brevity Penalty)
    bp = 1
    if len(cand_tokens) < len(ref_tokens):
        bp = math.exp(1 - len(ref_tokens)/len(cand_tokens))
        
    return precision * bp

def calculate_rouge_1_f1(candidate, reference):
    cand_tokens = candidate.split()
    ref_tokens = reference.split()
    matches = len(set(cand_tokens).intersection(set(ref_tokens)))
    
    recall = matches / len(ref_tokens) if ref_tokens else 0
    precision = matches / len(cand_tokens) if cand_tokens else 0
    
    if recall + precision == 0: return 0.0
    return 2 * (precision * recall) / (precision + recall)
```

---

**Q20: 实现 LLM-as-a-Judge 评估框架**
- 难度：⭐⭐⭐
- 时间：30分钟
- 语言：Python
- 标签：#评估 #LLMJudge

**【答案参考】**
```python
def llm_as_judge(judge_llm, question, answer, reference=None):
    prompt = f"[Question]\n{question}\n\n[Model Answer]\n{answer}\n\n"
    if reference:
        prompt += f"[Reference Answer]\n{reference}\n\n"
        
    prompt += "Please evaluate the model's answer. Give a score from 1 to 5 based on Accuracy and Helpfulness. Format output exactly as: Score: <number>\nReasoner: <your reason>"
    
    feedback = judge_llm(prompt)
    try:
        score_line = [line for line in feedback.split('\n') if 'Score:' in line][0]
        score = int(score_line.split(':')[1].strip())
    except Exception:
        score = -1
    return score, feedback
```

---

## 第六部分：综合实战 (由于偏向于系统设计组合，此部分留做组合思路展示)
- **Q22:** Mini RAG 可结合 Q11(切块) + Q1(向量化计算/API) + BM25查询组合完成。
- **Q23:** Agent 可结合 Q6(React), Q7(Registry) + Q8(DB Memory) 装配实现整个服务层。

---

## 第七部分：Transformer 核心组件手撕（11题）

**Q24: 手撕自注意力机制（用PyTorch）**
- 难度：⭐⭐⭐
-时间：30分钟

**【答案参考】**：同上面 Q1 的实现。


**Q25: 手撕自注意力机制（不用PyTorch）**
- 难度：⭐⭐⭐⭐
- 语言：Python + NumPy

**【答案参考】**
```python
import numpy as np

def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

class NumpySelfAttention:
    def __init__(self, d_model):
        self.d_model = d_model
        # 随机初始化权重
        self.Wq = np.random.randn(d_model, d_model)
        self.Wk = np.random.randn(d_model, d_model)
        self.Wv = np.random.randn(d_model, d_model)
        
    def forward(self, x, mask=None):
        Q = np.dot(x, self.Wq)
        K = np.dot(x, self.Wk)
        V = np.dot(x, self.Wv)
        
        # QxK.T: x维度是 (batch, seq, d_model) -> matmul需要对后两维
        scores = np.einsum('bmd,bnd->bmn', Q, K) / np.sqrt(self.d_model)
        
        if mask is not None:
            scores = np.where(mask == 0, -np.inf, scores)
            
        attn = softmax(scores, axis=-1)
        out = np.einsum('bmn,bnd->bmd', attn, V)
        return out
```

---

**Q26: 手撕多头注意力机制（用PyTorch）**
**【答案参考】**: 同上述 Q4 实现。 

---

**Q27: 手撕多头注意力机制（不用PyTorch）**
- 难度：⭐⭐⭐⭐⭐
- 标签：#MultiHeadAttention #NumPy

**【答案参考】**
```python
class NumpyMHA:
    def __init__(self, d_model, num_heads):
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.Wq = np.random.randn(d_model, d_model)
        self.Wk = np.random.randn(d_model, d_model)
        self.Wv = np.random.randn(d_model, d_model)
        self.Wo = np.random.randn(d_model, d_model)
        
    def forward(self, x):
        batch, seq, _ = x.shape
        Q = np.dot(x, self.Wq).reshape(batch, seq, self.num_heads, self.d_k).transpose(0,2,1,3)
        K = np.dot(x, self.Wk).reshape(batch, seq, self.num_heads, self.d_k).transpose(0,2,1,3)
        V = np.dot(x, self.Wv).reshape(batch, seq, self.num_heads, self.d_k).transpose(0,2,1,3)
        
        scores = np.einsum('bhmd,bhnd->bhmn', Q, K) / np.sqrt(self.d_k)
        attn = softmax(scores, axis=-1)
        out = np.einsum('bhmn,bhnd->bhmd', attn, V)
        
        # 拼接回 (batch, seq, d_model)
        out = out.transpose(0, 2, 1, 3).reshape(batch, seq, -1)
        return np.dot(out, self.Wo)
```

---

**Q28: 手撕MQA（Multi-Query Attention）**
- 难度：⭐⭐⭐⭐
- 语言：Python + PyTorch

**【答案参考】**
```python
import torch
import torch.nn as nn
import math

class MultiQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # MQA: Q有多头，K和V只有一个头！
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, self.d_k)
        self.W_v = nn.Linear(d_model, self.d_k)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        batch, seq, _ = x.shape
        Q = self.W_q(x).view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        # K, V 只有一个头: (batch, 1, seq, d_k)
        K = self.W_k(x).unsqueeze(1).transpose(2, 3) 
        V = self.W_v(x).unsqueeze(1)
        
        scores = torch.matmul(Q, K) / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        
        out = torch.matmul(attn, V) # broadcasting 机制会自动应用 1 个V 给所有 head 对应计算
        out = out.transpose(1, 2).contiguous().view(batch, seq, self.d_model)
        return self.W_o(out)
```

---

**Q29: 手撕绝对位置编码**
- 难度：⭐⭐⭐

**【答案参考】**
```python
def positional_encoding(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe # shape (max_len, d_model)
```

---

**Q30: 手撕RoPE（旋转位置编码）**
- 难度：⭐⭐⭐⭐⭐

**【答案参考】**
```python
class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, max_len=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_len).type_as(inv_freq)
        freqs = torch.einsum('i,j->ij', t, inv_freq) # shape (max_len, dim/2)
        
        # 复制两次形成 (max_len, dim)
        emb = torch.cat((freqs, freqs), dim=-1)
        # 存下 sin 和 cos 用于旋转
        self.register_buffer('cos_cached', emb.cos().unsqueeze(0).unsqueeze(0)) 
        self.register_buffer('sin_cached', emb.sin().unsqueeze(0).unsqueeze(0))
        
    def rotate_half(self, x):
        # 将 x 分成两半然后进行旋转: [-x2, x1]
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q, k, seq_len):
        # 截取对应序列长度的 sin cos
        cos = self.cos_cached[:, :, :seq_len, ...]
        sin = self.sin_cached[:, :, :seq_len, ...]
        
        # 旋转公式: (x * cos) + (rotate_half(x) * sin)
        q_rope = (q * cos) + (self.rotate_half(q) * sin)
        k_rope = (k * cos) + (self.rotate_half(k) * sin)
        return q_rope, k_rope
```

---

**Q31: 手撕Transformer中FFN代码**
- 难度：⭐⭐⭐

**【答案参考】**
```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # FFN(x) = max(0, xW1 + b1)W2 + b2
        return self.linear2(self.relu(self.linear1(x)))
```

---

**Q32: 手撕Layer Norm**
- 难度：⭐⭐⭐

**【答案参考】**
```python
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        
        out = (x - mean) / torch.sqrt(var + self.eps)
        return out * self.gamma + self.beta
```

---

**Q33: 手撕RMSNorm**
- 难度：⭐⭐⭐

**【答案参考】**
```python
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        # RMS = sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight
```

---

**Q34: 手撕FlashAttention（简化版）**
- 难度：⭐⭐⭐⭐⭐

**【答案参考】**（概念与循环分块模拟）
```python
def flash_attention_forward(Q, K, V, block_size=64):
    """
    Tiling 简化模拟：
    实际上 FlashAttention 是用 CUDA 直接控制 SRAM 编写的。此处是用 python for loop 的逻辑演示
    """
    batch, num_heads, seq_len, head_dim = Q.shape
    O = torch.zeros_like(Q)
    l = torch.zeros((batch, num_heads, seq_len, 1)) # normalizer
    m = torch.full((batch, num_heads, seq_len, 1), -float('inf')) # max值
    
    # 将 Q 分块在外层
    for i in range(0, seq_len, block_size):
        Qi = Q[:, :, i:i+block_size, :]
        Oi = O[:, :, i:i+block_size, :]
        mi = m[:, :, i:i+block_size, :]
        li = l[:, :, i:i+block_size, :]
        
        # 将 K, V 分块在内层
        for j in range(0, seq_len, block_size):
            Kj = K[:, :, j:j+block_size, :]
            Vj = V[:, :, j:j+block_size, :]
            
            # S = QK^T
            S_ij = torch.matmul(Qi, Kj.transpose(-2, -1)) / math.sqrt(head_dim)
            
            # 在线 softmax 计算
            m_ij = torch.max(S_ij, dim=-1, keepdim=True)[0]
            m_new = torch.max(mi, m_ij)
            
            P_ij = torch.exp(S_ij - m_new)
            
            # 更新归一化与权重
            l_new = torch.exp(mi - m_new) * li + torch.sum(P_ij, dim=-1, keepdim=True)
            
            # 更新输出
            Oi = (torch.exp(mi - m_new) * Oi + torch.matmul(P_ij, Vj))
            mi = m_new
            li = l_new
            
        # 写回
        O[:, :, i:i+block_size, :] = Oi / li
        m[:, :, i:i+block_size, :] = mi
        l[:, :, i:i+block_size, :] = li
        
    return O
```

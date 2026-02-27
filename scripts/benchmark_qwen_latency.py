#!/usr/bin/env python3
"""
Benchmark Qwen3-0.6B ASR Latency.
This script measures the time taken for single-shot transcription and 
simulated streaming transcription.
"""

import sys
import os
import time
import asyncio
import numpy as np
import logging
import argparse

# Add src to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audio.qwen_asr_local import QwenLocalASREngine
from src.config import AudioConfig

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("benchmark")

def generate_dummy_audio(duration_sec=3.0, sample_rate=16000):
    """Generate white noise as dummy audio."""
    num_samples = int(duration_sec * sample_rate)
    return np.random.uniform(-1.0, 1.0, num_samples).astype(np.float32)

async def run_benchmark(model_path, device, audio_duration=3.0, chunk_ms=500):
    print(f"\n🚀 Starting Benchmark...")
    print(f"📍 Model: {model_path}")
    print(f"📍 Device: {device}")
    print(f"📍 Audio Duration: {audio_duration}s")
    print("-" * 40)

    config = AudioConfig(
        engine_type="qwen_local",
        qwen_local_model_path=model_path,
        qwen_local_device=device
    )
    engine = QwenLocalASREngine(config)

    # 1. Measure Initialization
    t0 = time.perf_counter()
    await engine.initialize()
    init_time = time.perf_counter() - t0
    print(f"✅ Model Initialized in {init_time:.2f}s")

    # Generate test audio
    audio = generate_dummy_audio(audio_duration)
    
    # 2. Warmup
    await engine.transcribe(generate_dummy_audio(0.5))
    print(f"✅ Warmup complete")

    # 3. One-Shot Latency
    t0 = time.perf_counter()
    _ = await engine.transcribe(audio)
    oneshot_latency = time.perf_counter() - t0
    rtf = oneshot_latency / audio_duration
    print(f"\n📊 [One-Shot Results]")
    print(f"   Latency: {oneshot_latency:.4f}s")
    print(f"   RTF:     {rtf:.4f} (Real-Time Factor)")

    # 4. Simulated Streaming Latency
    print(f"\n📊 [Simulated Streaming (chunk={chunk_ms}ms)]")
    sr = 16000
    chunk_samples = int(chunk_ms / 1000 * sr)
    latencies = []
    
    # Split audio into chunks
    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i:i + chunk_samples]
        if len(chunk) < chunk_samples:
            break
            
        t0 = time.perf_counter()
        _ = await engine.transcribe(chunk)
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)
        
        # print(f"   Chunk {len(latencies):02d}: {elapsed:.4f}s")

    avg_lat = sum(latencies) / len(latencies)
    min_lat = min(latencies)
    max_lat = max(latencies)
    
    print(f"   Avg Latency: {avg_lat:.4f}s")
    print(f"   Min Latency: {min_lat:.4f}s")
    print(f"   Max Latency: {max_lat:.4f}s")
    print(f"   P95 (approx): {sorted(latencies)[int(0.95*len(latencies))]:.4f}s")

    print("-" * 40)
    print("Benchmark Finished.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="Qwen3-ASR-0.6B")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--chunk", type=int, default=500)
    args = parser.parse_args()

    # If model_path is relative, assume it's in the repo root
    if not os.path.isabs(args.model_path):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args.model_path = os.path.join(root, args.model_path)

    asyncio.run(run_benchmark(
        args.model_path, 
        args.device, 
        args.duration, 
        args.chunk
    ))

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(r"c:\Users\67091\Desktop\gjx_work\stealth-meeting-ai").absolute()))

from src.config import AppConfig

def test_config():
    config = AppConfig.from_env()
    print(f"LLM API Key starts with: {config.llm.api_key[:10]}...")
    print(f"LLM Base URL: {config.llm.base_url}")
    print(f"LLM Model: {config.llm.model}")
    
    # Check if NIM values match
    expected_url = "https://integrate.api.nvidia.com/v1"
    expected_model = "qwen/qwen3.5-397b-a17b"
    
    if config.llm.base_url == expected_url and config.llm.model == expected_model:
        print("\nConfiguration loaded successfully!")
    else:
        print("\nConfiguration mismatch!")
        sys.exit(1)

if __name__ == "__main__":
    test_config()

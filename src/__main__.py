"""
Entry point for running the meeting assistant as a module:
    python -m src.main
"""

import asyncio
from .main import main

if __name__ == "__main__":
    asyncio.run(main())

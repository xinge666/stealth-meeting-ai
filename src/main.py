"""
Unified entry point for the Stealth Meeting AI (Client-Server Architecture).
Usage:
    python -m src.main --mode server
    python -m src.main --mode client
"""

import argparse
import asyncio
import sys
import logging

def parse_args():
    parser = argparse.ArgumentParser(description="Stealth Meeting AI Launcher")
    parser.add_argument(
        "--mode", 
        choices=["server", "client"], 
        default="server",
        help="Run as 'server' (heavy compute) or 'client' (audio/vision capture)"
    )
    return parser.parse_args()

async def main():
    args = parse_args()
    
    if args.mode == "server":
        print("🚀 Starting Compute Server...")
        from .server.main import main as server_main
        await server_main()
    else:
        print("🎙️  Starting Edge Client...")
        from .client.main import main as client_main
        await client_main()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)

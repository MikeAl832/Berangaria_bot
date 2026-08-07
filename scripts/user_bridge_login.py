#!/usr/bin/env python3
"""One-time login for the read-only user bridge.

Prints a Telethon StringSession for USER_BRIDGE_SESSION (.env / GitHub Secrets).
Never commit the printed value.

Usage (from repo root, with TELEGRAM_API_ID and TELEGRAM_API_HASH in .env):

    python scripts/user_bridge_login.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Repo root on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")


async def main() -> int:
    api_id = (os.environ.get("TELEGRAM_API_ID") or "").strip()
    api_hash = (os.environ.get("TELEGRAM_API_HASH") or "").strip()
    if not api_id or not api_hash:
        print("Set TELEGRAM_API_ID and TELEGRAM_API_HASH in .env first.", file=sys.stderr)
        return 1
    try:
        api_id_int = int(api_id)
    except ValueError:
        print("TELEGRAM_API_ID must be an integer.", file=sys.stderr)
        return 1

    try:
        from telethon import TelegramClient
        from telethon.sessions import StringSession
    except ImportError:
        print("Install telethon: pip install telethon", file=sys.stderr)
        return 1

    print("Logging in for user-bridge (read-only listener).")
    print("You will receive a login code in the official Telegram app.\n")

    client = TelegramClient(StringSession(), api_id_int, api_hash)
    await client.start()
    session_str = client.session.save()
    me = await client.get_me()
    await client.disconnect()

    print("\n--- SUCCESS ---")
    print(f"Logged in as: {getattr(me, 'username', None) or me.first_name} (id={me.id})")
    print("\nAdd this to .env (and GitHub Actions secret USER_BRIDGE_SESSION):\n")
    print(f"USER_BRIDGE_SESSION={session_str}")
    print("\nAlso set in config.yaml: user_bridge_enabled: true")
    print("Do NOT commit this value to git.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

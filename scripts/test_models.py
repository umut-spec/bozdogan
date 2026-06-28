"""Test different models"""
import asyncio
import aiohttp
import json
import os
from dotenv import load_dotenv

load_dotenv()

async def test_models():
    API_KEY = os.getenv("CICI_API_KEY")

    models = [
        "gemini-2.0-flash-exp",
        "gemini-exp-1206",
        "gpt-4o",
        "claude-3-5-sonnet-20241022"
    ]

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    for model in models:
        print(f"\n{'='*50}")
        print(f"Testing: {model}")
        print('='*50)

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Merhaba"}],
            "max_tokens": 50
        }

        try:
            timeout = aiohttp.ClientTimeout(total=20)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    "https://api.ciciapi.com/v1/chat/completions",
                    json=payload,
                    headers=headers
                ) as response:
                    print(f"Status: {response.status}")
                    if response.status == 200:
                        data = await response.json()
                        print(f"SUCCESS! Model works: {model}")
                        return model  # İlk çalışan modeli döndür
                    else:
                        text = await response.text()
                        print(f"Failed: {text[:100]}")

        except Exception as e:
            print(f"Error: {type(e).__name__}")

        await asyncio.sleep(2)

    return None

if __name__ == "__main__":
    working_model = asyncio.run(test_models())
    if working_model:
        print(f"\n\nCALISAN MODEL: {working_model}")
    else:
        print("\n\nHIC BIR MODEL CALISMADI!")

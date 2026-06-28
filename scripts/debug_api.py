"""Quick debug test"""
import asyncio
import aiohttp
import json
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("CICI_API_KEY")
API_ENDPOINT = "https://api.ciciapi.com/v1/chat/completions"

async def test_single_request():
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gemini-2.0-flash",
        "messages": [{"role": "user", "content": "Merhaba, test mesajı"}],
        "temperature": 0.9,
        "max_tokens": 100
    }

    print(f"API Key ilk 10 karakter: {API_KEY[:10] if API_KEY else 'YOK!'}...")
    print(f"Endpoint: {API_ENDPOINT}")
    print(f"Request gonderiliyor...\n")

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(API_ENDPOINT, json=payload, headers=headers, timeout=30) as response:
                print(f"Status: {response.status}")
                print(f"Headers: {dict(response.headers)}\n")

                text = await response.text()
                print(f"Response body:\n{text[:500]}\n")

                if response.status == 200:
                    data = json.loads(text)
                    print(f"SUCCESS! Content: {data['choices'][0]['message']['content'][:100]}")
                else:
                    print(f"FAILED! Status {response.status}")
        except Exception as e:
            print(f"ERROR: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(test_single_request())

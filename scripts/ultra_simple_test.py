"""Ultra basit test - tek istek"""
import asyncio
import aiohttp
import json
import os
from dotenv import load_dotenv

load_dotenv()

async def simple_test():
    API_KEY = os.getenv("CICI_API_KEY")

    print(f"API Key: {API_KEY[:15]}...")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gemini-2.0-flash",
        "messages": [{"role": "user", "content": "Bana 3 tane basit soru-cevap yaz JSON formatında: [{'user': 'soru', 'assistant': 'cevap'}]"}],
        "temperature": 0.7,
        "max_tokens": 500
    }

    print("\nIstek gonderiliyor...")

    try:
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                "https://api.ciciapi.com/v1/chat/completions",
                json=payload,
                headers=headers
            ) as response:
                print(f"Status Code: {response.status}")
                text = await response.text()
                print(f"\nResponse:\n{text[:300]}\n")

                if response.status == 200:
                    data = json.loads(text)
                    print("BASARILI!")
                    print(data["choices"][0]["message"]["content"])
                else:
                    print(f"HATA: {response.status}")

    except Exception as e:
        print(f"EXCEPTION: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(simple_test())

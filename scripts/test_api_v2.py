"""
CiciAPI bağlantısını test et (HTTP requests ile)
"""

import os
import sys
import requests
import json
from dotenv import load_dotenv

# Windows terminal için encoding ayarı
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

def test_connection():
    """API bağlantısını test et"""

    api_key = os.getenv("CICI_API_KEY")

    if not api_key:
        print("[X] HATA: CICI_API_KEY bulunamadi!")
        print("Lutfen .env dosyasinda CICI_API_KEY tanimlayin")
        return False

    print("[OK] API Key bulundu")
    print(f"     Key: {api_key[:8]}...")

    try:
        url = "https://api.ciciapi.com/v1/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        payload = {
            "model": "gemini-3-flash",
            "messages": [
                {"role": "user", "content": "Merhaba, bu bir test mesaji. 'Test basarili' diye cevap ver."}
            ],
            "max_tokens": 50
        }

        print("\n[>] API'ye baglaniliyor...")

        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()

        data = response.json()
        content = data['choices'][0]['message']['content']

        print("[OK] Baglanti basarili!")
        print(f"\n[<] Model cevabi:")
        print(f"    {content}\n")

        # Kullanım bilgileri
        if 'usage' in data:
            usage = data['usage']
            print(f"[i] Token kullanimi:")
            print(f"    Input: {usage.get('prompt_tokens', 0)}")
            print(f"    Output: {usage.get('completion_tokens', 0)}")
            print(f"    Total: {usage.get('total_tokens', 0)}")

        return True

    except requests.exceptions.RequestException as e:
        print(f"[X] HATA: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"[i] Status code: {e.response.status_code}")
            try:
                print(f"[i] Response: {e.response.json()}")
            except:
                print(f"[i] Response text: {e.response.text}")
        return False
    except Exception as e:
        print(f"[X] HATA: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("CiciAPI Baglanti Testi")
    print("=" * 60)
    print()

    success = test_connection()

    if success:
        print("\n[OK] API kullanima hazir!")
        print("      Simdi veri uretimini baslat:")
        print("      python src/data_generation/generate_synthetic_data.py")
    else:
        print("\n[X] API baglantisi basarisiz!")
        print("    Lutfen .env dosyanizi kontrol edin")

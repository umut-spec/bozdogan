"""
CiciAPI bağlantısını test et
"""

import os
import sys
from openai import OpenAI
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
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.ciciapi.com"
        )

        print("\n[>] API'ye baglaniliyor...")

        response = client.chat.completions.create(
            model="gemini-3-flash",
            messages=[
                {"role": "user", "content": "Hello, this is a test message. Please respond with 'Test successful'."}
            ],
            max_tokens=50,
            timeout=30
        )

        content = response.choices[0].message.content

        print("[OK] Baglanti basarili!")
        print(f"\n[<] Model cevabi:")
        print(f"    {content}\n")

        # Kullanım bilgileri
        if hasattr(response, 'usage'):
            print(f"[i] Token kullanimi:")
            print(f"    Input: {response.usage.prompt_tokens}")
            print(f"    Output: {response.usage.completion_tokens}")
            print(f"    Total: {response.usage.total_tokens}")

        return True

    except Exception as e:
        print(f"[X] HATA: {e}")
        print(f"[i] Hata tipi: {type(e).__name__}")
        if hasattr(e, 'response'):
            print(f"[i] Response: {e.response}")
        if hasattr(e, 'status_code'):
            print(f"[i] Status code: {e.status_code}")
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

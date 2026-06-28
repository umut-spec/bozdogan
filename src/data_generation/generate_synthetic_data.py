"""
Gemini 3 Flash ile sentetik Türkçe konuşma verisi üretimi
CiciAPI + requests kullanımı
"""

import os
import sys
import json
import time
import requests
from typing import List, Dict
from tqdm import tqdm
from dotenv import load_dotenv

# Windows terminal için encoding ayarı
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

class SyntheticDataGenerator:
    def __init__(self):
        self.api_key = os.getenv("CICI_API_KEY")
        self.base_url = "https://api.ciciapi.com/v1/chat/completions"
        self.model = "gemini-3-flash"

        if not self.api_key:
            raise ValueError("CICI_API_KEY bulunamadı! .env dosyasını kontrol edin.")

    def generate_batch(self, num_conversations: int = 20) -> List[Dict]:
        """
        Tek bir API çağrısında birden fazla konuşma örneği üret

        Args:
            num_conversations: Üretilecek konuşma sayısı

        Returns:
            List of conversation dictionaries
        """

        system_prompt = """Sen bir Türkçe dil uzmanısın. Görevin yüksek kaliteli, doğal ve çeşitli Türkçe konuşma örnekleri üretmek.

Her konuşma şu formatta olmalı:
{
  "user": "kullanıcı mesajı",
  "assistant": "asistan cevabı"
}

Konuşmalar şunları içermeli:
- Günlük hayat konuları (alışveriş, yemek, seyahat, teknoloji, vb.)
- Farklı zorluk seviyeleri (basit - karmaşık)
- Farklı konuşma tarzları (resmi - samimi)
- Türkçe'ye özgü deyimler ve ifadeler
- Doğal ve akıcı dil kullanımı

ÖNEMLİ: Sadece JSON array dön, başka açıklama yapma."""

        user_prompt = f"""Lütfen {num_conversations} farklı Türkçe konuşma örneği üret.

Konular çeşitli olsun:
- Günlük sohbet
- Teknik sorular
- Yardım talepleri
- Bilgi sorgulamaları
- Problem çözme
- Tavsiye isteme

JSON array formatında dön:
[
  {{"user": "...", "assistant": "..."}},
  {{"user": "...", "assistant": "..."}}
]"""

        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.9,
                "max_tokens": 4000
            }

            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()

            data = response.json()
            content = data['choices'][0]['message']['content'].strip()

            # JSON parse et
            # Bazen markdown code block içinde geliyor, temizle
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]

            conversations = json.loads(content.strip())

            return conversations

        except requests.exceptions.RequestException as e:
            print(f"\nHTTP Hatası: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Status: {e.response.status_code}")
                try:
                    print(f"Response: {e.response.json()}")
                except:
                    print(f"Response text: {e.response.text[:200]}")
            return []
        except json.JSONDecodeError as e:
            print(f"\nJSON Parse Hatası: {e}")
            print(f"Content: {content[:200]}...")
            return []
        except Exception as e:
            print(f"\nGenel Hata: {e}")
            return []

    def generate_dataset(
        self,
        num_requests: int = 1000,
        conversations_per_request: int = 20,
        output_file: str = "data/raw/synthetic_data.json",
        checkpoint_interval: int = 100
    ):
        """
        Toplu veri üretimi

        Args:
            num_requests: API istek sayısı
            conversations_per_request: İstek başına konuşma sayısı
            output_file: Çıktı dosyası
            checkpoint_interval: Her kaç istekte bir kaydet
        """

        all_conversations = []
        total_generated = 0

        print(f"Hedef: {num_requests} istek × {conversations_per_request} konuşma = {num_requests * conversations_per_request} örnek")
        print(f"Tahmini maliyet: {num_requests * 0.010} yuan\n")

        for i in tqdm(range(num_requests), desc="Veri üretiliyor"):
            batch = self.generate_batch(conversations_per_request)

            if batch:
                all_conversations.extend(batch)
                total_generated += len(batch)

            # Checkpoint kaydet
            if (i + 1) % checkpoint_interval == 0:
                self._save_checkpoint(all_conversations, output_file, i + 1)
                print(f"\n[OK] Checkpoint: {total_generated} ornek kaydedildi")

            # Rate limiting (isteğe bağlı)
            time.sleep(0.1)

        # Final kaydet
        self._save_data(all_conversations, output_file)

        print(f"\n[OK] Tamamlandi!")
        print(f"Toplam uretilen: {len(all_conversations)} konusma")
        print(f"Dosya: {output_file}")

        return all_conversations

    def _save_checkpoint(self, data: List[Dict], output_file: str, request_num: int):
        """Checkpoint kaydet"""
        checkpoint_file = output_file.replace(".json", f"_checkpoint_{request_num}.json")
        self._save_data(data, checkpoint_file)

    def _save_data(self, data: List[Dict], output_file: str):
        """Veriyi kaydet"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # .env dosyasında CICI_API_KEY olmalı

    generator = SyntheticDataGenerator()

    # 1000 istek × 20 konuşma = 20,000 örnek
    # Maliyet: 10 yuan (~$1.4)
    dataset = generator.generate_dataset(
        num_requests=1000,
        conversations_per_request=20,
        output_file="data/raw/synthetic_data.json",
        checkpoint_interval=100
    )

"""
Gemini 3 Flash ile sentetik Türkçe konuşma verisi üretimi (PARALEL)
CiciAPI + requests + concurrent.futures
"""

import os
import sys
import json
import time
import requests
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv

# Windows terminal için encoding ayarı
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()


class ParallelSyntheticDataGenerator:
    def __init__(self, max_workers: int = 50):
        self.api_key = os.getenv("CICI_API_KEY")
        self.base_url = "https://api.ciciapi.com/v1/chat/completions"
        self.model = "gemini-3-flash"
        self.max_workers = max_workers

        if not self.api_key:
            raise ValueError("CICI_API_KEY bulunamadi! .env dosyasini kontrol edin.")

    def generate_batch(self, batch_id: int, num_conversations: int = 20) -> tuple:
        """
        Tek bir API çağrısında birden fazla konuşma örneği üret
        Returns: (batch_id, conversations_list)
        """

        system_prompt = """Sen bir Türkçe dil uzmanısın. Görevin yüksek kaliteli, doğal ve çeşitli Türkçe konuşma örnekleri üretmek.

Her konuşma şu formatta olmalı:
{
  "user": "kullanıcı mesajı",
  "assistant": "asistan cevabı"
}

Konuşmalar şunları içermeli:
- Günlük hayat konuları (alışveriş, yemek, seyahat, teknoloji, eğitim, sağlık, iş, hobi)
- Farklı zorluk seviyeleri (basit - karmaşık)
- Farklı konuşma tarzları (resmi - samimi)
- Türkçe'ye özgü deyimler ve ifadeler
- Doğal ve akıcı dil kullanımı

ÖNEMLİ: Sadece JSON array dön, başka açıklama yapma."""

        user_prompt = f"""Lütfen {num_conversations} farklı Türkçe konuşma örneği üret.

ÖNEMLİ: Asistan cevapları UZUN ve DETAYLI olmalı (minimum 200-500 karakter).

Konular çeşitli olsun:
- Günlük sohbet ve hal hatır (detaylı hikayeler)
- Teknik sorular ve açıklamalar (adım adım anlatım)
- Yardım talepleri ve öneriler (birden fazla seçenek sun)
- Bilgi sorgulamaları (kapsamlı açıklamalar)
- Problem çözme ve troubleshooting (detaylı çözüm yolları)
- Tavsiye isteme ve verme (örneklerle zenginleştir)
- Alışveriş ve rezervasyon (detaylı ürün/yer açıklamaları)
- Tarif ve yönlendirme (adım adım detaylı anlatım)
- Öğrenme ve eğitim (kavramları derinlemesine açıkla)
- Sağlık ve spor (detaylı program ve öneriler)
- Finans ve yatırım (rakamlar ve örneklerle açıkla)
- Seyahat ve tatil (çok detaylı rota ve öneriler)

Asistan cevapları:
- Minimum 200-500 karakter olmalı
- Detaylı açıklamalar içermeli
- Örnekler, listeler, adımlar vermeli
- Zengin kelime dağarcığı kullanmalı
- Paragraflar halinde düzenli olmalı

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
                "max_tokens": 8000  # 4000'den 8000'e çıkardık (daha uzun cevaplar için)
            }

            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=90  # Timeout'u artırdık
            )

            # Rate limit için kısa bekle
            time.sleep(0.1)

            response.raise_for_status()

            data = response.json()
            content = data['choices'][0]['message']['content'].strip()

            # JSON parse et
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]

            conversations = json.loads(content.strip())

            return (batch_id, conversations)

        except requests.exceptions.HTTPError as e:
            # 503 hatası için özel mesaj
            if e.response.status_code == 503:
                print(f"\n[!] Batch {batch_id}: API mesgul (503), tekrar denenecek...")
                time.sleep(2)
                # Bir kez daha dene
                try:
                    response = requests.post(
                        self.base_url,
                        headers=headers,
                        json=payload,
                        timeout=90
                    )
                    if response.status_code == 200:
                        data = response.json()
                        content = data['choices'][0]['message']['content'].strip()
                        if content.startswith("```json"):
                            content = content[7:]
                        if content.startswith("```"):
                            content = content[3:]
                        if content.endswith("```"):
                            content = content[:-3]
                        conversations = json.loads(content.strip())
                        return (batch_id, conversations)
                except:
                    pass
            return (batch_id, [])
        except Exception as e:
            # Hata durumunda batch_id ile boş liste dön
            return (batch_id, [])

    def generate_dataset_parallel(
        self,
        num_requests: int = 1000,
        conversations_per_request: int = 20,
        output_file: str = "data/raw/synthetic_data.json",
        checkpoint_interval: int = 100
    ):
        """
        Paralel veri üretimi

        Args:
            num_requests: API istek sayısı
            conversations_per_request: İstek başına konuşma sayısı
            output_file: Çıktı dosyası
            checkpoint_interval: Her kaç istekte bir kaydet
        """

        print(f"Hedef: {num_requests} istek x {conversations_per_request} konusma = {num_requests * conversations_per_request} ornek")
        print(f"Tahmini maliyet: {num_requests * 0.010} yuan")
        print(f"Max paralel istek: {self.max_workers}")
        print()

        all_conversations = []
        failed_batches = []

        # ThreadPoolExecutor ile paralel istekler
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Tüm istekleri submit et
            future_to_batch = {
                executor.submit(self.generate_batch, i, conversations_per_request): i
                for i in range(num_requests)
            }

            # Progress bar
            with tqdm(total=num_requests, desc="Veri uretiliyor") as pbar:
                for future in as_completed(future_to_batch):
                    batch_id, conversations = future.result()

                    if conversations:
                        all_conversations.extend(conversations)
                    else:
                        failed_batches.append(batch_id)

                    pbar.update(1)

                    # Checkpoint kaydet
                    if (pbar.n) % checkpoint_interval == 0:
                        self._save_checkpoint(all_conversations, output_file, pbar.n)
                        print(f"\n[OK] Checkpoint: {len(all_conversations)} ornek kaydedildi (istek: {pbar.n}/{num_requests})")

        # Final kaydet
        self._save_data(all_conversations, output_file)

        print(f"\n[OK] Tamamlandi!")
        print(f"Toplam uretilen: {len(all_conversations)} konusma")
        print(f"Basarili istek: {num_requests - len(failed_batches)}/{num_requests}")
        if failed_batches:
            print(f"Basarisiz istek: {len(failed_batches)}")
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
    # Test
    generator = ParallelSyntheticDataGenerator(max_workers=50)

    # 10 istek ile test (paralel)
    dataset = generator.generate_dataset_parallel(
        num_requests=10,
        conversations_per_request=20,
        output_file="data/raw/parallel_test.json",
        checkpoint_interval=5
    )

"""
MEGA DATASET GENERATOR
10,000 istek, saniyede 100 paralel, tüm kategoriler
"""

import asyncio
import aiohttp
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict
from tqdm.asyncio import tqdm
from dotenv import load_dotenv

load_dotenv()

# Kategoriler ve prompt'lar
CATEGORIES = {
    "teknoloji": [
        "Python programlama hakkında pratik sorular",
        "Web geliştirme teknikleri",
        "Yapay zeka ve makine öğrenmesi",
        "Siber güvenlik ve etik hacking",
        "Mobil uygulama geliştirme",
        "Veri bilimi ve analitik",
        "Bulut teknolojileri (AWS, Azure, GCP)",
        "DevOps ve CI/CD",
        "Blockchain ve kripto",
        "IoT ve akıllı cihazlar"
    ],
    "günlük_hayat": [
        "Yemek tarifleri ve mutfak ipuçları",
        "Sağlıklı yaşam ve beslenme",
        "Ev düzeni ve organizasyon",
        "Alışveriş ve bütçe yönetimi",
        "İlişkiler ve sosyal durumlar",
        "Evcil hayvan bakımı",
        "Hobi ve el işleri",
        "Ev onarımı ve tadilat",
        "Çocuk bakımı ve ebeveynlik",
        "Kişisel bakım ve hijyen"
    ],
    "eğitim": [
        "Matematik problemleri ve açıklamalar",
        "Tarih ve coğrafya bilgileri",
        "Dil öğrenimi (İngilizce, Almanca vs.)",
        "Sınav hazırlık teknikleri",
        "Üniversite ve kariyer tavsiyeleri",
        "Bilimsel kavramlar (fizik, kimya, biyoloji)",
        "Edebiyat ve kitap önerileri",
        "Ödev yardımı ve kaynak önerileri",
        "Online eğitim platformları",
        "Araştırma ve makale yazımı"
    ],
    "iş_kariyer": [
        "Mülakat hazırlığı ve CV yazımı",
        "Kariyer değişikliği tavsiyeleri",
        "İş yerinde iletişim",
        "Freelance çalışma ve uzaktan iş",
        "Girişimcilik ve startup kurmak",
        "Zaman yönetimi ve verimlilik",
        "Networking ve profesyonel gelişim",
        "Maaş pazarlığı ve terfi stratejileri",
        "İş hukuku ve çalışan hakları",
        "Liderlik ve ekip yönetimi"
    ],
    "eğlence": [
        "Film ve dizi önerileri",
        "Müzik türleri ve sanatçılar",
        "Oyun önerileri (video, masa, kağıt)",
        "Seyahat yerleri ve gezi rotaları",
        "Spor ve fitness aktiviteleri",
        "Konser ve etkinlik tavsiyeleri",
        "Fotoğrafçılık ve video çekimi",
        "Sosyal medya trendleri",
        "Podcast ve YouTube kanalları",
        "Koleksiyon ve hobi dünyası"
    ],
    "sağlık": [
        "Grip ve soğuk algınlığı tedavisi",
        "Stres ve anksiyete yönetimi",
        "Uyku düzeni ve kalitesi",
        "Egzersiz ve fitness programları",
        "Diyet ve kilo yönetimi",
        "Ruh sağlığı ve terapi",
        "İlk yardım bilgileri",
        "Kronik hastalık yönetimi",
        "Alternatif tıp ve bitkisel tedavi",
        "Medikal tetkikler ve tahliller"
    ],
    "finans": [
        "Bütçe planlama ve tasarruf",
        "Yatırım stratejileri (hisse, altın, döviz)",
        "Kredi kartı ve borç yönetimi",
        "Emeklilik planlaması",
        "Kripto para yatırımları",
        "Sigorta türleri ve seçimi",
        "Vergi ve muhasebe bilgileri",
        "Gayrimenkul yatırımı",
        "Finansal okuryazarlık",
        "Ekonomik haberler ve analiz"
    ],
    "hukuk": [
        "Tüketici hakları",
        "Kira ve emlak hukuku",
        "İş hukuku ve işçi hakları",
        "Aile hukuku (boşanma, nafaka)",
        "Ticaret ve şirket hukuku",
        "Ceza hukuku temelleri",
        "İnsan hakları",
        "Sosyal medya ve dijital haklar",
        "Miras ve vasiyet işlemleri",
        "Trafik suçları ve cezaları"
    ]
}

API_ENDPOINT = "https://api.ciciapi.com/v1/chat/completions"


class MassiveDatasetGenerator:
    def __init__(self, total_requests: int = 10000, concurrent_limit: int = 100):
        self.total_requests = total_requests
        self.concurrent_limit = concurrent_limit
        self.api_key = os.getenv("CICI_API_KEY")
        self.results = []
        self.failed_requests = []
        self.retry_limit = 3  # Her istek için max 3 deneme

        # Output dizini
        self.output_dir = Path("data/raw")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = self.output_dir / f"massive_dataset_{timestamp}.json"
        self.checkpoint_file = self.output_dir / f"massive_checkpoint_{timestamp}.json"

    def create_prompt(self, category: str, topic: str) -> str:
        """Her kategori için özel prompt oluştur"""
        return f"""Sen Türkçe konuşan bir yapay zeka asistanısın. '{category}' kategorisinde '{topic}' konusunda 10 farklı kullanıcı-asistan diyalogu üret.

KURALLAR:
1. Her diyalog gerçekçi ve doğal olmalı
2. Kullanıcı mesajları günlük konuşma dilinde olmalı
3. Asistan cevapları bilgilendirici, yardımcı ve samimi olmalı
4. Farklı soru tipleri kullan (nasıl, neden, ne, önerir misin, vs.)
5. Bazı diyaloglarda argo/konuşma dili, bazılarında resmi dil kullan
6. Cevap uzunlukları çeşitli olsun (kısa-orta-uzun)

JSON formatında döndür:
[
  {{"user": "kullanıcı mesajı", "assistant": "asistan cevabı"}},
  ...
]

SADECE JSON array döndür, başka açıklama ekleme."""

    async def generate_single(
        self,
        session: aiohttp.ClientSession,
        category: str,
        topic: str,
        request_num: int
    ) -> List[Dict]:
        """Tek bir API isteği yap (retry logic ile)"""
        prompt = self.create_prompt(category, topic)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "gemini-2.0-flash",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.9,
            "max_tokens": 4000
        }

        # Retry logic
        for attempt in range(self.retry_limit):
            try:
                async with session.post(API_ENDPOINT, json=payload, headers=headers, timeout=60) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data["choices"][0]["message"]["content"]

                        # JSON parse
                        content = content.strip()
                        if content.startswith("```json"):
                            content = content[7:]
                        if content.endswith("```"):
                            content = content[:-3]
                        content = content.strip()

                        examples = json.loads(content)

                        # Metadata ekle
                        for ex in examples:
                            ex["category"] = category
                            ex["topic"] = topic
                            ex["request_num"] = request_num

                        return examples

                    elif response.status == 503:
                        # Service busy - retry sonra
                        if attempt < self.retry_limit - 1:
                            await asyncio.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                            continue
                        else:
                            error_msg = "503: Service Busy (tried 3 times)"
                            self.failed_requests.append({
                                "request_num": request_num,
                                "category": category,
                                "topic": topic,
                                "error": error_msg
                            })
                            return []

                    else:
                        error_text = await response.text()
                        error_msg = f"HTTP {response.status}: {error_text[:200]}"
                        self.failed_requests.append({
                            "request_num": request_num,
                            "category": category,
                            "topic": topic,
                            "error": error_msg
                        })
                        # İlk 5 hatayı print et
                        if len(self.failed_requests) <= 5:
                            print(f"\n[HATA] Request {request_num}: {error_msg}\n")
                        return []

            except asyncio.TimeoutError:
                if attempt < self.retry_limit - 1:
                    await asyncio.sleep(2)
                    continue
                error_msg = "Timeout (60s, tried 3 times)"
                self.failed_requests.append({
                    "request_num": request_num,
                    "category": category,
                    "topic": topic,
                    "error": error_msg
                })
                if len(self.failed_requests) <= 5:
                    print(f"\n[HATA] Request {request_num}: {error_msg}\n")
                return []
            except Exception as e:
                if attempt < self.retry_limit - 1 and "json" not in str(e).lower():
                    await asyncio.sleep(1)
                    continue
                error_msg = f"{type(e).__name__}: {str(e)[:200]}"
                self.failed_requests.append({
                    "request_num": request_num,
                    "category": category,
                    "topic": topic,
                    "error": error_msg
                })
                if len(self.failed_requests) <= 5:
                    print(f"\n[HATA] Request {request_num}: {error_msg}\n")
                return []

        return []

    async def generate_batch(self, batch_requests: List[tuple]) -> List[Dict]:
        """Bir batch request'i paralel çalıştır"""
        async with aiohttp.ClientSession() as session:
            tasks = [
                self.generate_single(session, cat, topic, req_num)
                for req_num, cat, topic in batch_requests
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Exception'ları filtrele
            valid_results = []
            for r in results:
                if isinstance(r, list):
                    valid_results.extend(r)

            return valid_results

    def save_checkpoint(self):
        """Checkpoint kaydet"""
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)

        print(f"\n[CHECKPOINT] {len(self.results)} ornek kaydedildi: {self.checkpoint_file}")

    async def generate_all(self):
        """Tüm veriyi üret"""
        print("=" * 70)
        print("MASSIVE DATASET GENERATOR")
        print("=" * 70)
        print(f"Toplam istek: {self.total_requests:,}")
        print(f"Paralel limit: {self.concurrent_limit}")
        print(f"Tahmini ornek: ~{self.total_requests * 10:,}")
        print(f"Output: {self.output_file}")
        print("=" * 70)

        # İstekleri hazırla (kategori ve topic'leri dengeli dağıt)
        all_requests = []
        categories_list = list(CATEGORIES.items())

        for i in range(self.total_requests):
            # Döngüsel olarak kategorileri kullan
            cat_idx = i % len(categories_list)
            category, topics = categories_list[cat_idx]

            # Topic'i de döngüsel seç
            topic_idx = (i // len(categories_list)) % len(topics)
            topic = topics[topic_idx]

            all_requests.append((i + 1, category, topic))

        # Batch'lere böl
        batches = [
            all_requests[i:i + self.concurrent_limit]
            for i in range(0, len(all_requests), self.concurrent_limit)
        ]

        print(f"\nToplam {len(batches)} batch halinde islenecek")
        print(f"Her batch: {self.concurrent_limit} paralel istek")
        print(f"\nBaslangic: {datetime.now().strftime('%H:%M:%S')}\n")

        # Progress bar ile batch'leri işle
        for batch_idx, batch in enumerate(tqdm(batches, desc="Batches")):
            batch_results = await self.generate_batch(batch)
            self.results.extend(batch_results)

            # Batch'ler arası rate limit için kısa bekle
            await asyncio.sleep(0.5)

            # Her 10 batch'te bir checkpoint
            if (batch_idx + 1) % 10 == 0:
                self.save_checkpoint()

                # İstatistikler
                print(f"\n--- Ilerleme Raporu ---")
                print(f"Tamamlanan istek: {(batch_idx + 1) * self.concurrent_limit:,} / {self.total_requests:,}")
                print(f"Toplam ornek: {len(self.results):,}")
                print(f"Basarisiz istek: {len(self.failed_requests)}")
                print(f"Ortalama ornek/istek: {len(self.results) / ((batch_idx + 1) * self.concurrent_limit):.1f}")
                print(f"-" * 40 + "\n")

        # Final checkpoint
        self.save_checkpoint()

        # Sonuç raporu
        print("\n" + "=" * 70)
        print("TAMAMLANDI!")
        print("=" * 70)
        print(f"Toplam ornek: {len(self.results):,}")
        print(f"Basarili istek: {self.total_requests - len(self.failed_requests):,}")
        print(f"Basarisiz istek: {len(self.failed_requests)}")
        print(f"Basari orani: {((self.total_requests - len(self.failed_requests)) / self.total_requests * 100):.1f}%")
        print(f"\nOutput dosyasi: {self.output_file}")

        # Kategori dağılımı
        print(f"\n--- Kategori Dagilimi ---")
        category_counts = {}
        for item in self.results:
            cat = item.get("category", "unknown")
            category_counts[cat] = category_counts.get(cat, 0) + 1

        for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"{cat:20s}: {count:6,} ornek ({count/len(self.results)*100:5.1f}%)")

        # Final save
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)

        # Failed requests kaydet
        if self.failed_requests:
            failed_file = self.output_dir / f"failed_requests_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(failed_file, 'w', encoding='utf-8') as f:
                json.dump(self.failed_requests, f, ensure_ascii=False, indent=2)
            print(f"\nBasarisiz istekler: {failed_file}")

        print("=" * 70)


async def main():
    generator = MassiveDatasetGenerator(
        total_requests=10000,      # 10K istek
        concurrent_limit=50        # 50 paralel (100'den düştük, rate limit için)
    )

    await generator.generate_all()


if __name__ == "__main__":
    asyncio.run(main())

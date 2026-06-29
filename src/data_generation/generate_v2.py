"""
Bozdoğan sentetik veri üretimi v2 — yüksek kalite + çeşitlilik
- Domain / persona / stil / zorluk rotasyonu (mode collapse'a karşı)
- Çok-turlu (multi-turn) konuşmalar
- Zengin prompt + kalite rubriği + few-shot
- İstek başına AZ ama DOLU örnek (kalite > nicelik)
"""

import os
import sys
import json
import time
import random
import requests
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

# --- Çeşitlilik havuzları: her istek bunlardan rastgele kombinasyon seçer ---

DOMAINS = [
    "yemek ve mutfak", "seyahat ve rota planı", "teknoloji ve yazılım",
    "sağlık ve beslenme", "spor ve antrenman", "kişisel finans ve bütçe",
    "kariyer ve iş görüşmesi", "eğitim ve sınav hazırlığı", "psikoloji ve motivasyon",
    "tarih ve kültür", "bilim ve doğa", "sanat ve müzik", "kitap ve film önerisi",
    "ev düzeni ve tamir", "bahçecilik", "ebeveynlik ve çocuk gelişimi",
    "hukuk ve günlük haklar (genel bilgi)", "girişimcilik ve pazarlama",
    "dil öğrenme", "felsefe ve etik tartışma", "oyun ve eğlence",
    "araç bakımı", "moda ve stil", "evcil hayvan bakımı",
    "iklim ve çevre", "astronomi", "matematik problemi çözümü",
    "yazım ve metin düzenleme", "münazara ve ikna", "günlük hal hatır sohbeti",
]

PERSONAS = [
    "meraklı bir üniversite öğrencisi", "zaman sıkıntısı olan bir profesyonel",
    "teknolojiye yeni başlayan biri", "iki çocuklu bir ebeveyn",
    "emekli ve sakin bir kullanıcı", "konuyu derinlemesine kurcalayan bir uzman",
    "kararsız ve seçenek isteyen biri", "bütçesi kısıtlı bir kullanıcı",
    "aceleci ve net cevap isteyen biri", "duygusal destek arayan biri",
]

STYLES = [
    "resmi ve kibar", "samimi ve sıcak", "esprili ama bilgilendirici",
    "kısa ve öz", "detaylı ve adım adım", "sokak ağzına yakın doğal Türkçe",
]

DIFFICULTIES = [
    "basit günlük", "orta seviye", "ileri/uzmanlık gerektiren", "çok adımlı problem",
]

TURN_STYLES = [
    "tek soru-tek cevap",
    "2 turlu (kullanıcı takip sorusu sorar)",
    "3 turlu (kullanıcı detaylandırır ve itiraz eder)",
]


SYSTEM_PROMPT = """Sen Türkçe konuşma verisi üreten kıdemli bir veri mühendisisin. \
Bir Türkçe yapay zeka asistanını eğitmek için yüksek kaliteli, gerçekçi diyaloglar üretiyorsun.

KALİTE KURALLARI (hepsine uy):
1. Türkçe dil bilgisi kusursuz olmalı: ş, ı, ğ, ç, ö, ü, İ harfleri doğru kullanılmalı.
2. Asistan cevapları DOĞRU, işe yarar ve somut olmalı. Uydurma bilgi (halüsinasyon) verme.
3. Cevaplar yapılandırılmış olsun: gerektiğinde madde işaretleri, adımlar, kısa örnekler.
4. Asistan kibar ama yapmacık değil; gereksiz dolgu cümle ("Tabii ki!", "Harika soru!") ile başlamasın.
5. Kullanıcı mesajları gerçek insan gibi: bazen yazım hatalı, kısa, bağlamlı, doğal olsun.
6. Çok turlu diyaloglarda asistan önceki turu hatırlasın, tutarlı kalsın.
7. Hassas konularda (sağlık, hukuk, finans) genel bilgi ver ve "uzmana danış" uyarısı ekle.

ÇIKTI: SADECE geçerli JSON array dön. Markdown, açıklama, ```json bloğu YOK."""


# Modelin formatı öğrenmesi için 1 adet kaliteli few-shot örnek
FEWSHOT = """[
  {
    "turns": [
      {"role": "user", "content": "Evde kahve yapıyorum ama hep acı oluyor, ne yapabilirim?"},
      {"role": "assistant", "content": "Acılığın birkaç tipik sebebi var, sırayla bakalım:\\n\\n1. Su sıcaklığı: Kaynar su kahveyi yakar. İdeal aralık 90-96°C. Kaynattıktan sonra 30 saniye bekle.\\n2. Demleme süresi: French press için 4 dakika yeterli; daha uzunu acılaştırır.\\n3. Öğütüm: Çok ince çekim aşırı ekstraksiyona yol açar. Demleme yöntemine göre kalınlaştır.\\n\\nÖnce su sıcaklığını düşürmeyi dene, çoğu zaman tek başına çözüyor."},
      {"role": "user", "content": "French press kullanıyorum, öğütümü nasıl ayarlamalıyım?"},
      {"role": "assistant", "content": "French press için kaba öğütüm gerekir; iri deniz tuzu kıvamında düşün. İnce çekim hem acılaştırır hem de pistondan tortu kaçırır. Değirmenin varsa en kaba 2-3 kademede dene, sonuca göre ince ayar yap."}
    ]
  }
]"""


def build_user_prompt(num_convos: int) -> str:
    """Her istek için rastgele kombinasyon seçerek çeşitlilik zorlar."""
    specs = []
    for i in range(num_convos):
        specs.append(
            f"{i+1}. Konu: {random.choice(DOMAINS)} | "
            f"Kullanıcı: {random.choice(PERSONAS)} | "
            f"Üslup: {random.choice(STYLES)} | "
            f"Zorluk: {random.choice(DIFFICULTIES)} | "
            f"Yapı: {random.choice(TURN_STYLES)}"
        )
    spec_block = "\n".join(specs)

    return f"""Aşağıdaki {num_convos} spesifikasyonun HER BİRİ için bir diyalog üret. \
Her diyalog kendi satırındaki konu/persona/üslup/zorluk/yapıya UYMALI:

{spec_block}

Format (örnek kalite seviyesi):
{FEWSHOT}

Kurallar:
- Her öğe {{"turns": [{{"role": "user/assistant", "content": "..."}}]}} yapısında olsun.
- "tek soru-tek cevap" ise 2 turn (1 user + 1 assistant), "2 turlu" ise 4 turn, "3 turlu" ise 6 turn olsun.
- Asistan cevapları dolu olsun ama gevezelik değil; her cümle bilgi taşısın.
- SADECE JSON array dön."""


def _strip_fences(content: str) -> str:
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    return content.strip()


class QualityDataGenerator:
    def __init__(self, max_workers: int = 30, convos_per_request: int = 6):
        self.api_key = os.getenv("CICI_API_KEY")
        self.base_url = "https://api.ciciapi.com/v1/chat/completions"
        self.model = "gemini-3-flash"
        self.max_workers = max_workers
        self.convos_per_request = convos_per_request
        if not self.api_key:
            raise ValueError("CICI_API_KEY bulunamadı! .env dosyasını kontrol edin.")

    def _post(self, payload: dict, timeout: int = 120) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        resp = requests.post(self.base_url, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def generate_batch(self, batch_id: int, max_retries: int = 5) -> tuple:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(self.convos_per_request)},
            ],
            "temperature": 1.0,   # çeşitlilik için yüksek
            "top_p": 0.95,
            "max_tokens": 8000,
        }
        for attempt in range(max_retries):
            try:
                content = self._post(payload)
                items = json.loads(_strip_fences(content))
                return (batch_id, self._to_records(items))
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response is not None else None
                if status == 429:   # rate limit — varsa Retry-After'a uy, yoksa kademeli bekle
                    ra = e.response.headers.get("Retry-After", "") if e.response is not None else ""
                    wait_s = float(ra) if ra.replace(".", "", 1).isdigit() else min(2 ** attempt * 2, 30)
                    time.sleep(wait_s)
                else:                # 5xx vb. sunucu hatası
                    time.sleep(2 * (attempt + 1))
            except requests.exceptions.Timeout:
                time.sleep(2 * (attempt + 1))
            except json.JSONDecodeError:
                time.sleep(1)   # bozuk JSON, tekrar dene
            except Exception:
                time.sleep(1)
        return (batch_id, [])

    @staticmethod
    def _fingerprint(record: Dict) -> int:
        """Aynı diyaloğun tekrar kaydedilmesini önlemek için içerik parmak izi."""
        turns = record.get("turns", [])
        return hash("|".join((t.get("content") or "") for t in turns))

    def _to_records(self, items: List[Dict]) -> List[Dict]:
        """v2 çıktısını ({"turns":[...]}) standart kayıtlara çevir + doğrula."""
        records = []
        for it in items:
            turns = it.get("turns") if isinstance(it, dict) else None
            if not turns or len(turns) < 2:
                continue
            # rol sırası user ile başlamalı, dönüşümlü olmalı
            roles = [t.get("role") for t in turns]
            if roles[0] != "user" or len(turns) % 2 != 0:
                continue
            clean_turns = []
            ok = True
            for idx, t in enumerate(turns):
                expected = "user" if idx % 2 == 0 else "assistant"
                content = (t.get("content") or "").strip()
                if t.get("role") != expected or len(content) < 5:
                    ok = False
                    break
                clean_turns.append({"role": expected, "content": content})
            if ok:
                records.append({"turns": clean_turns})
        return records

    def generate(self, output_file: str, max_requests: int = 0, checkpoint_interval: int = 50):
        """max_requests=0 ise SINIRSIZ: Ctrl+C'ye kadar üretir, durunca kaydeder.

        Var olan output_file okunur ve üzerine devam edilir (dedup ile).
        """
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

        all_records: List[Dict] = []
        seen = set()
        if os.path.exists(output_file):           # önceki çalışmadan devam et
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    all_records = json.load(f)
                for r in all_records:
                    seen.add(self._fingerprint(r))
                print(f"Mevcut dosyadan devam: {len(all_records)} diyalog yüklendi.")
            except (json.JSONDecodeError, OSError):
                print("Mevcut dosya okunamadı, sıfırdan başlanıyor.")

        limit_txt = "SINIRSIZ (Ctrl+C ile durdur)" if max_requests == 0 else f"{max_requests} istek"
        print(f"Mod: {limit_txt} | worker: {self.max_workers} | istek başına {self.convos_per_request} diyalog\n")

        sent = dups = failed = 0
        stop = False

        def _enough() -> bool:
            return max_requests != 0 and sent >= max_requests

        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                # Havuzu doldur
                futures = set()
                for _ in range(self.max_workers):
                    if _enough():
                        break
                    futures.add(ex.submit(self.generate_batch, sent)); sent += 1

                with tqdm(total=(max_requests or None), desc="Kaliteli veri", unit="istek") as pbar:
                    while futures:
                        done = next(as_completed(futures))
                        futures.remove(done)
                        _, recs = done.result()
                        if recs:
                            for r in recs:                       # dedup
                                fp = self._fingerprint(r)
                                if fp in seen:
                                    dups += 1
                                    continue
                                seen.add(fp)
                                all_records.append(r)
                        else:
                            failed += 1
                        pbar.update(1)
                        pbar.set_postfix(diyalog=len(all_records), dup=dups, hata=failed)

                        if pbar.n % checkpoint_interval == 0:
                            self._save(all_records, output_file)

                        # Bitmediyse yeni iş ekle (sürekli besle)
                        if not stop and not _enough():
                            futures.add(ex.submit(self.generate_batch, sent)); sent += 1
        except KeyboardInterrupt:
            stop = True
            print("\n[!] Durduruldu — toplanan veri kaydediliyor...")

        self._save(all_records, output_file)
        print(f"\n[OK] {len(all_records)} benzersiz diyalog | tekrar atlanan: {dups} | başarısız istek: {failed}")
        print(f"Dosya: {output_file}")
        return all_records

    def _save(self, data: List[Dict], path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-requests", type=int, default=0,
                        help="API istek sayısı (0 = SINIRSIZ, Ctrl+C ile durur)")
    parser.add_argument("--convos", type=int, default=6, help="İstek başına diyalog")
    parser.add_argument("--workers", type=int, default=30)
    parser.add_argument("--out", type=str, default="data/raw/synthetic_v2.json")
    args = parser.parse_args()

    gen = QualityDataGenerator(max_workers=args.workers, convos_per_request=args.convos)
    gen.generate(output_file=args.out, max_requests=args.max_requests)



"""
KİMLİK VERİSİ + MEVCUT VERİYİ KARIŞTIR
Bozdoğan'a kimliğini öğretirken Türkçe yeteneğini korur
"""

import json
import jsonlines
import random
import sys
from pathlib import Path

sys.path.append('src')
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from data_processing.process_data import DataProcessor

print("="*60)
print("🦅 BOZDOĞAN KİMLİK EĞİTİM SETİ HAZIRLAMA")
print("="*60)
print()

# 1. Kimlik verisini yükle
with open("data/raw/bozdogan_identity.json", 'r', encoding='utf-8') as f:
    identity_data = json.load(f)
print(f"[OK] Kimlik verisi: {len(identity_data)} örnek")

# 2. Kimlik verisini oversample et (5x tekrar - iyice öğrensin)
identity_oversampled = identity_data * 5
print(f"[OK] Kimlik oversample (5x): {len(identity_oversampled)} örnek")

# 3. Mevcut train verisinden örnek al (unutmasın diye)
existing_data = []
with jsonlines.open("data/splits/train.jsonl") as reader:
    for item in reader:
        existing_data.append({"user": item["user"], "assistant": item["assistant"]})

# Mevcut veriden rastgele 2000 örnek al (hepsini almaya gerek yok, hız için)
random.shuffle(existing_data)
existing_sample = existing_data[:2000]
print(f"[OK] Mevcut veriden örnek: {len(existing_sample)} örnek")

# 4. Karıştır
combined = identity_oversampled + existing_sample
random.shuffle(combined)
print(f"[OK] Toplam karışık veri: {len(combined)} örnek")
print()

# 5. Birleştirilmiş veriyi kaydet
combined_file = Path("data/raw/bozdogan_identity_combined.json")
with open(combined_file, 'w', encoding='utf-8') as f:
    json.dump(combined, f, ensure_ascii=False, indent=2)

# 6. İşle ve split'le
print("Veri işleniyor...")
processor = DataProcessor(str(combined_file))
processor.process_pipeline(
    model_type="qwen",
    output_dir="data/splits_identity"
)

print()
print("="*60)
print("[TAMAMLANDI!] Kimlik eğitim seti hazır: data/splits_identity/")
print("="*60)

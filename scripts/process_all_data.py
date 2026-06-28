"""
TÜM HAM VERİLERİ BİRLEŞTİR VE İŞLE
"""

import json
import sys
from pathlib import Path

sys.path.append('src')
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from data_processing.process_data import DataProcessor

# Ham veri dosyaları
raw_files = [
    "data/raw/synthetic_data.json",
    "data/raw/parallel_test.json",
    "data/raw/test_mega.json",
    "data/raw/test_synthetic_data.json"
]

print("=" * 70)
print("TUM HAM VERILERI BIRLESTIR VE ISLE")
print("=" * 70)
print()

# Tüm verileri yükle
all_data = []
for file_path in raw_files:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                all_data.extend(data)
                print(f"[OK] {file_path}: {len(data)} ornek yuklendi")
            else:
                print(f"[ATLANDI] {file_path}: List degil")
    except FileNotFoundError:
        print(f"[ATLANDI] {file_path}: Dosya bulunamadi")
    except Exception as e:
        print(f"[HATA] {file_path}: {e}")

print()
print(f"Toplam yuklenen: {len(all_data)} ornek")
print()

# Birleştirilmiş veriyi kaydet
combined_file = "data/raw/all_combined_data.json"
with open(combined_file, 'w', encoding='utf-8') as f:
    json.dump(all_data, f, ensure_ascii=False, indent=2)

print(f"[OK] Birlestirilmis veri kaydedildi: {combined_file}")
print()

# İşleme pipeline'ı
print("=" * 70)
print("VERI ISLEME BASLADI")
print("=" * 70)
print()

processor = DataProcessor(combined_file)

processor.process_pipeline(
    model_type="qwen",
    output_dir="data/splits"
)

print()
print("=" * 70)
print("[TAMAMLANDI!] TUM VERILER ISLENDI!")
print("=" * 70)

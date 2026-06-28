"""
TÜM HAM VERİLERİ BİRLEŞTİR VE İŞLE
En güncel checkpoint dosyalarını kullanır ve duplikatları temizler
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.append('src')
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from data_processing.process_data import DataProcessor

def find_latest_checkpoints(raw_dir: Path):
    """Her veri serisi için en güncel checkpoint'i bul"""

    checkpoint_series = defaultdict(list)

    # Tüm JSON dosyalarını tara
    for json_file in raw_dir.glob("*.json"):
        name = json_file.stem

        # Checkpoint numarasını ayıkla
        if "_checkpoint_" in name:
            base_name = name.split("_checkpoint_")[0]
            checkpoint_num = int(name.split("_checkpoint_")[1])
            checkpoint_series[base_name].append((checkpoint_num, json_file))
        else:
            # Checkpoint içermeyen dosyalar (orijinal veriler)
            checkpoint_series[name].append((0, json_file))

    # Her seri için en yüksek checkpoint'i seç
    latest_files = []
    for base_name, checkpoints in checkpoint_series.items():
        checkpoints.sort(key=lambda x: x[0], reverse=True)
        latest_checkpoint_num, latest_file = checkpoints[0]
        latest_files.append((base_name, latest_checkpoint_num, latest_file))

    return latest_files


def load_and_combine_data(files_info):
    """Dosyalardan veriyi yükle ve birleştir"""
    all_data = []
    total_loaded = 0

    print("=" * 70)
    print("HAM VERİ YÜKLEME")
    print("=" * 70)
    print()

    for base_name, checkpoint_num, file_path in files_info:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, list):
                count = len(data)
                all_data.extend(data)
                total_loaded += count

                checkpoint_info = f"checkpoint_{checkpoint_num}" if checkpoint_num > 0 else "original"
                print(f"[OK] {base_name} ({checkpoint_info}): {count:,} örnek")
            else:
                print(f"[ATLANDI] {file_path.name}: List değil")

        except FileNotFoundError:
            print(f"[ATLANDI] {file_path.name}: Dosya bulunamadı")
        except json.JSONDecodeError as e:
            print(f"[HATA] {file_path.name}: JSON parse hatası - {e}")
        except Exception as e:
            print(f"[HATA] {file_path.name}: {e}")

    print()
    print(f"Toplam yüklenen: {total_loaded:,} örnek")
    print(f"Ham toplam: {len(all_data):,} örnek")

    return all_data


def remove_duplicates(data):
    """Duplikat örnekleri temizle"""
    print()
    print("=" * 70)
    print("DUPLİKAT TEMİZLEME")
    print("=" * 70)
    print()

    seen = set()
    unique_data = []

    for item in data:
        # Her örnek için unique key oluştur
        user_msg = item.get("user", "").strip()
        assistant_msg = item.get("assistant", "").strip()

        key = f"{user_msg}|{assistant_msg}"

        if key not in seen:
            seen.add(key)
            unique_data.append(item)

    removed = len(data) - len(unique_data)
    print(f"Ham örnek: {len(data):,}")
    print(f"Benzersiz örnek: {len(unique_data):,}")
    print(f"Temizlenen duplikat: {removed:,}")

    return unique_data


if __name__ == "__main__":
    # Raw data klasörü
    raw_dir = Path("data/raw")

    # 1. En güncel checkpoint'leri bul
    print("=" * 70)
    print("EN GÜNCEL CHECKPOINT'LERİ BULMA")
    print("=" * 70)
    print()

    latest_files = find_latest_checkpoints(raw_dir)

    print(f"Toplam {len(latest_files)} veri serisi bulundu:")
    for base_name, checkpoint_num, _ in sorted(latest_files, key=lambda x: x[0]):
        if checkpoint_num > 0:
            print(f"  - {base_name}: checkpoint_{checkpoint_num}")
        else:
            print(f"  - {base_name}: original")
    print()

    # 2. Verileri yükle ve birleştir
    all_data = load_and_combine_data(latest_files)

    # 3. Duplikatları temizle
    unique_data = remove_duplicates(all_data)

    # 4. Birleştirilmiş veriyi kaydet
    combined_file = Path("data/processed/combined_raw_data.json")
    combined_file.parent.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 70)
    print("BİRLEŞTİRİLMİŞ VERİYİ KAYDETME")
    print("=" * 70)
    print()

    with open(combined_file, 'w', encoding='utf-8') as f:
        json.dump(unique_data, f, ensure_ascii=False, indent=2)

    print(f"[OK] Birleştirilmiş veri kaydedildi: {combined_file}")
    print(f"[OK] Toplam: {len(unique_data):,} benzersiz örnek")
    print()

    # 5. İşleme pipeline'ı başlat
    print("=" * 70)
    print("VERİ İŞLEME PIPELINE")
    print("=" * 70)
    print()

    processor = DataProcessor(str(combined_file))

    processor.process_pipeline(
        model_type="qwen",
        output_dir="data/splits"
    )

    print()
    print("=" * 70)
    print("[TAMAMLANDI!] TÜM VERİLER İŞLENDİ!")
    print("=" * 70)
    print()
    print(f"Çıktılar:")
    print(f"  - data/processed/combined_raw_data.json ({len(unique_data):,} örnek)")
    print(f"  - data/splits/train.jsonl")
    print(f"  - data/splits/val.jsonl")
    print(f"  - data/splits/test.jsonl")

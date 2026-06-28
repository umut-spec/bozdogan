"""
TÜM SİSTEMİ KONTROL ET VE EĞİTİME HAZIRLIK DURUMUNU GÖSTER
"""

import sys
import subprocess
from pathlib import Path
import json
import jsonlines

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def check_files():
    """Gerekli dosyaları kontrol et"""
    print("=" * 70)
    print("📁 DOSYA KONTROLÜ")
    print("=" * 70)
    print()

    required_files = [
        ("configs/training_config.yaml", "Training konfigürasyonu"),
        ("src/training/train.py", "Training scripti"),
        ("data/splits/train.jsonl", "Training verisi"),
        ("data/splits/val.jsonl", "Validation verisi"),
        ("data/splits/test.jsonl", "Test verisi"),
    ]

    all_exist = True

    for file_path, description in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"✅ {description}: {file_path}")
        else:
            print(f"❌ {description}: {file_path} - BULUNAMADI!")
            all_exist = False

    print()
    return all_exist


def check_data_quality():
    """Veri kalitesini kontrol et"""
    print("=" * 70)
    print("📊 VERİ KALİTESİ KONTROLÜ")
    print("=" * 70)
    print()

    splits = ["train", "val", "test"]
    stats = {}

    for split in splits:
        file_path = Path(f"data/splits/{split}.jsonl")

        if not file_path.exists():
            print(f"❌ {split}.jsonl bulunamadı!")
            continue

        # Yükle
        data = []
        with jsonlines.open(file_path) as reader:
            for item in reader:
                data.append(item)

        stats[split] = len(data)

        # Format kontrolü
        has_required_fields = all('text' in item and 'user' in item and 'assistant' in item for item in data)

        if has_required_fields:
            print(f"✅ {split:5s}: {len(data):,} örnek - Format OK")
        else:
            print(f"⚠️  {split:5s}: {len(data):,} örnek - Format HATALI")

    print()
    print(f"Toplam veri: {sum(stats.values()):,} örnek")
    print()

    return stats


def check_python_packages():
    """Python paketlerini kontrol et"""
    print("=" * 70)
    print("📦 PYTHON PAKET KONTROLÜ")
    print("=" * 70)
    print()

    required_packages = [
        "torch",
        "transformers",
        "peft",
        "bitsandbytes",
        "accelerate",
        "trl",
        "datasets",
    ]

    all_installed = True

    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - KURULU DEĞİL!")
            all_installed = False

    print()
    return all_installed


def check_gpu():
    """GPU kontrolü"""
    print("=" * 70)
    print("🖥️  GPU KONTROLÜ")
    print("=" * 70)
    print()

    try:
        import torch

        if torch.cuda.is_available():
            print(f"✅ CUDA mevcut: {torch.version.cuda}")
            print(f"✅ GPU sayısı: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")

            # VRAM check
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if gpu_memory >= 24:
                print(f"✅ VRAM yeterli: {gpu_memory:.1f} GB")
            else:
                print(f"⚠️  VRAM düşük: {gpu_memory:.1f} GB (Minimum 24GB önerilir)")
        else:
            print("❌ CUDA mevcut değil!")
            print("⚠️  CPU ile eğitim çok yavaş olacak!")
            print("💡 Vast.ai veya cloud GPU kullanmanız önerilir")

        print()

    except ImportError:
        print("❌ PyTorch kurulu değil!")
        print()


def show_training_commands():
    """Eğitim komutlarını göster"""
    print("=" * 70)
    print("🚀 EĞİTİM KOMUTLARI")
    print("=" * 70)
    print()

    print("Eğitimi başlatmak için:")
    print()
    print("  python src/training/train.py")
    print()
    print("Veya özel konfigürasyon ile:")
    print()
    print("  python src/training/train.py --config configs/training_config.yaml")
    print()
    print("TensorBoard ile izleme:")
    print()
    print("  tensorboard --logdir models/qwen-turkish-chat")
    print()


def show_summary(files_ok, packages_ok, data_stats):
    """Özet göster"""
    print("=" * 70)
    print("📋 ÖZET")
    print("=" * 70)
    print()

    if files_ok and packages_ok and data_stats:
        print("✅ SİSTEM TAM HAZIR!")
        print()
        print(f"  • Veri: {sum(data_stats.values()):,} örnek")
        print(f"  • Train: {data_stats.get('train', 0):,} örnek")
        print(f"  • Val: {data_stats.get('val', 0):,} örnek")
        print(f"  • Test: {data_stats.get('test', 0):,} örnek")
        print()
        print("Eğitime başlayabilirsiniz! 🎉")
    else:
        print("⚠️  HAZIRLIK EKSİK")
        print()
        if not files_ok:
            print("  • Bazı dosyalar eksik")
        if not packages_ok:
            print("  • Bazı paketler kurulu değil")
            print("    Çözüm: pip install -r requirements.txt")
        if not data_stats:
            print("  • Veri dosyaları eksik")
            print("    Çözüm: python scripts/process_raw_data.py")

    print()


if __name__ == "__main__":
    print()
    print("🔍 SİSTEM HAZIRLIK KONTROLÜ")
    print()

    # Kontroller
    files_ok = check_files()
    data_stats = check_data_quality()
    packages_ok = check_python_packages()
    check_gpu()
    show_training_commands()
    show_summary(files_ok, packages_ok, data_stats)

    print("=" * 70)
    print()

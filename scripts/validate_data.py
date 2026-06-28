"""
Eğitim verilerini doğrula ve istatistikleri göster
"""

import json
import jsonlines
from pathlib import Path
from collections import Counter
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def load_jsonl(file_path):
    """JSONL dosyasını yükle"""
    data = []
    with jsonlines.open(file_path) as reader:
        for item in reader:
            data.append(item)
    return data


def validate_format(data, split_name):
    """Format doğrulama"""
    print(f"\n{'='*60}")
    print(f"{split_name.upper()} DOĞRULAMA")
    print(f"{'='*60}")

    errors = []

    # Required fields
    required_fields = ['text', 'user', 'assistant']

    for i, item in enumerate(data):
        # Field kontrolü
        for field in required_fields:
            if field not in item:
                errors.append(f"Örnek {i}: '{field}' alanı eksik")

        # Boş kontrol
        if item.get('text', '').strip() == '':
            errors.append(f"Örnek {i}: 'text' alanı boş")

        # Format kontrolü (Qwen)
        text = item.get('text', '')
        if '<|im_start|>' not in text or '<|im_end|>' not in text:
            errors.append(f"Örnek {i}: Qwen chat formatı eksik")

    if errors:
        print(f"\n⚠️  {len(errors)} hata bulundu:")
        for error in errors[:10]:  # İlk 10 hatayı göster
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... ve {len(errors) - 10} hata daha")
    else:
        print(f"✅ Format doğrulaması: OK")

    return len(errors) == 0


def show_statistics(data, split_name):
    """İstatistikleri göster"""

    # Uzunluk istatistikleri
    text_lengths = [len(item['text']) for item in data]
    user_lengths = [len(item['user']) for item in data]
    assistant_lengths = [len(item['assistant']) for item in data]

    print(f"\n📊 {split_name.upper()} İSTATİSTİKLER")
    print(f"{'─'*60}")
    print(f"Toplam örnek: {len(data):,}")
    print()
    print(f"Text uzunluğu:")
    print(f"  Min: {min(text_lengths):,} karakter")
    print(f"  Max: {max(text_lengths):,} karakter")
    print(f"  Ortalama: {sum(text_lengths) // len(text_lengths):,} karakter")
    print()
    print(f"User mesaj uzunluğu:")
    print(f"  Min: {min(user_lengths):,} karakter")
    print(f"  Max: {max(user_lengths):,} karakter")
    print(f"  Ortalama: {sum(user_lengths) // len(user_lengths):,} karakter")
    print()
    print(f"Assistant mesaj uzunluğu:")
    print(f"  Min: {min(assistant_lengths):,} karakter")
    print(f"  Max: {max(assistant_lengths):,} karakter")
    print(f"  Ortalama: {sum(assistant_lengths) // len(assistant_lengths):,} karakter")


def show_sample(data, split_name, n=2):
    """Örnek göster"""
    print(f"\n📝 {split_name.upper()} ÖRNEK ({n} adet)")
    print(f"{'─'*60}")

    for i in range(min(n, len(data))):
        item = data[i]
        print(f"\nÖrnek #{i+1}:")
        print(f"User: {item['user'][:100]}...")
        print(f"Assistant: {item['assistant'][:100]}...")


if __name__ == "__main__":
    splits_dir = Path("data/splits")

    print("="*60)
    print("EĞİTİM VERİSİ DOĞRULAMA")
    print("="*60)

    # Her split için doğrulama
    all_valid = True

    for split_name in ["train", "val", "test"]:
        file_path = splits_dir / f"{split_name}.jsonl"

        if not file_path.exists():
            print(f"\n❌ {split_name}.jsonl bulunamadı!")
            all_valid = False
            continue

        # Yükle
        data = load_jsonl(file_path)

        # Doğrula
        valid = validate_format(data, split_name)
        all_valid = all_valid and valid

        # İstatistikler
        show_statistics(data, split_name)

        # Örnekler
        show_sample(data, split_name, n=1)

    # Final sonuç
    print(f"\n{'='*60}")
    if all_valid:
        print("✅ TÜM VERİLER DOĞRULANDI - EĞİTİME HAZIR!")
    else:
        print("❌ HATALAR VAR - LÜTFEN DÜZELTİN!")
    print(f"{'='*60}")

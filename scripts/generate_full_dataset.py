"""
Tam veri üretimi - 1000 istek × 20 konuşma = 20,000 örnek
"""

import sys
sys.path.append('src')

# Windows encoding fix
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from data_generation.generate_synthetic_data import SyntheticDataGenerator

if __name__ == "__main__":
    print("=" * 60)
    print("TAM VERI URETIMI")
    print("=" * 60)
    print()

    generator = SyntheticDataGenerator()

    print("Hedef: 1000 istek x 20 konusma = 20,000 ornek")
    print("Tahmini maliyet: 10 yuan (~$1.4)")
    print("Tahmini sure: 25-30 dakika")
    print()
    print("Checkpoint'ler her 100 istekte kaydedilir.")
    print("Kesinti durumunda checkpoint'ten devam edebilirsiniz.")
    print()
    print("Baslaniyor...")
    print()

    dataset = generator.generate_dataset(
        num_requests=1000,
        conversations_per_request=20,
        output_file="data/raw/synthetic_data.json",
        checkpoint_interval=100
    )

    print(f"\n[OK] TAMAMLANDI!")
    print(f"Toplam uretilen: {len(dataset)} konusma")
    print(f"Dosya: data/raw/synthetic_data.json")
    print()
    print("Sonraki adim:")
    print("python src/data_processing/process_data.py")

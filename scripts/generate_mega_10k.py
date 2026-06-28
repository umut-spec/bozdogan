"""
MEGA PARALEL VERİ ÜRETİMİ - 10,000 istek
Her istek 10 örnek = 100,000 toplam örnek hedefi
"""

import sys
sys.path.append('src')

# Windows encoding fix
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from data_generation.generate_parallel import ParallelSyntheticDataGenerator

if __name__ == "__main__":
    print("=" * 70)
    print("MEGA PARALEL VERI URETIMI - 10K ISTEK")
    print("=" * 70)
    print()
    print("Hedef: 10,000 istek x 20 ornek = 200,000 ornek")
    print("Paralel worker: 30 (rate limit icin dusuruldu)")
    print("Checkpoint: Her 500 istek")
    print()
    print("=" * 70)
    print()

    # 30 paralel worker (rate limit için düşük tuttuk)
    generator = ParallelSyntheticDataGenerator(max_workers=30)

    print("BASLIYOR...")
    print()

    dataset = generator.generate_dataset_parallel(
        num_requests=10000,              # 10K istek
        conversations_per_request=20,    # Her istekte 20 örnek (10'dan 20'ye çıkardık)
        output_file="data/raw/mega_synthetic_data.json",
        checkpoint_interval=500          # Her 500 istekte checkpoint
    )

    print()
    print("=" * 70)
    print("[TAMAMLANDI!] MEGA DATASET OLUSTURULDU!")
    print("=" * 70)
    print()
    print(f"Toplam ornek: {len(dataset):,}")
    print(f"Output: data/raw/mega_synthetic_data.json")
    print()
    print("Sonraki adim:")
    print("python src/data_processing/process_data.py")

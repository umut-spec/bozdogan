"""
TAM PARALEL VERİ ÜRETİMİ - 1000 istek
"""

import sys
sys.path.append('src')

# Windows encoding fix
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from data_generation.generate_parallel import ParallelSyntheticDataGenerator

if __name__ == "__main__":
    print("=" * 60)
    print("TAM PARALEL VERI URETIMI")
    print("=" * 60)
    print()

    # 50 paralel worker (daha fazla artırabilirsiniz)
    generator = ParallelSyntheticDataGenerator(max_workers=50)

    print("BASLIYOR...")
    print()

    dataset = generator.generate_dataset_parallel(
        num_requests=1000,
        conversations_per_request=20,
        output_file="data/raw/synthetic_data.json",
        checkpoint_interval=100
    )

    print()
    print("=" * 60)
    print("[OK] TAMAMLANDI!")
    print("=" * 60)
    print()
    print("Sonraki adim:")
    print("python src/data_processing/process_data.py")

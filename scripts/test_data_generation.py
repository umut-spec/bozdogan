"""
Test script - 5 istekle veri üretimini test et
"""

import sys
sys.path.append('src')

# Windows encoding fix
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from data_generation.generate_synthetic_data import SyntheticDataGenerator

if __name__ == "__main__":
    print("=" * 60)
    print("VERI URETIMI TESTI")
    print("=" * 60)
    print()

    generator = SyntheticDataGenerator()

    print("Test: 5 istek x 10 konusma = 50 ornek")
    print("Tahmini maliyet: 0.05 yuan (~$0.007)")
    print()

    dataset = generator.generate_dataset(
        num_requests=5,
        conversations_per_request=10,
        output_file="data/raw/test_synthetic_data.json",
        checkpoint_interval=2
    )

    print(f"\n[OK] Test tamamlandi!")
    print(f"  Uretilen: {len(dataset)} konusma")

    if dataset:
        print(f"\n[i] Ilk ornek:")
        print(f"  User: {dataset[0]['user'][:100]}...")
        print(f"  Assistant: {dataset[0]['assistant'][:100]}...")

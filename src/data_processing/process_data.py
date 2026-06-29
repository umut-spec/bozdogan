"""
Ham sentetik veriyi fine-tuning için işle
"""

import json
import jsonlines
import random
from pathlib import Path
from typing import List, Dict
from datasets import Dataset
from tqdm import tqdm


class DataProcessor:
    def __init__(self, input_files):
        # Tek dosya da liste de kabul et
        self.input_files = [input_files] if isinstance(input_files, str) else list(input_files)
        self.data = self._load_data()

    def _load_data(self) -> List[Dict]:
        """Bir veya birden çok ham dosyayı yükle, tek formata indirge ({"turns": [...]})."""
        normalized = []
        for path in self.input_files:
            with open(path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            for item in raw:
                # Yeni v2 formatı: {"turns": [{"role","content"}, ...]}
                if isinstance(item, dict) and "turns" in item:
                    normalized.append(item)
                # Eski format: {"user": ..., "assistant": ...}
                elif isinstance(item, dict) and "user" in item and "assistant" in item:
                    normalized.append({"turns": [
                        {"role": "user", "content": item["user"]},
                        {"role": "assistant", "content": item["assistant"]},
                    ]})
            print(f"  yuklendi: {path}")
        return normalized

    def clean_data(self) -> List[Dict]:
        """
        Veriyi temizle ve filtrele (çok-turlu destekli)
        - Boş/çok kısa/çok uzun turn'leri ele
        - Rol sırası doğrulaması (user/assistant dönüşümlü)
        - Duplikatları temizle
        """
        cleaned = []
        seen = set()

        print("Veri temizleniyor...")

        for item in tqdm(self.data):
            turns = item.get("turns", [])
            if len(turns) < 2 or len(turns) % 2 != 0:
                continue

            valid = True
            for idx, t in enumerate(turns):
                expected = "user" if idx % 2 == 0 else "assistant"
                content = (t.get("content") or "").strip()
                if t.get("role") != expected:
                    valid = False
                    break
                # Uzunluk kontrolü
                if expected == "user" and not (5 <= len(content) <= 2000):
                    valid = False
                    break
                if expected == "assistant" and not (10 <= len(content) <= 4000):
                    valid = False
                    break
            if not valid:
                continue

            # Duplikat kontrolü (tüm diyalog metni üzerinden)
            key = "|".join((t.get("content") or "").strip() for t in turns)
            if key in seen:
                continue

            seen.add(key)
            cleaned.append({"turns": [
                {"role": t["role"], "content": t["content"].strip()} for t in turns
            ]})

        print(f"[OK] Temizleme: {len(self.data)} -> {len(cleaned)} ornek")
        return cleaned

    def format_for_training(self, data: List[Dict], model_type: str = "qwen") -> List[Dict]:
        """
        Modele uygun format'a çevir (çok-turlu)

        Args:
            data: Temiz veri ({"turns": [...]})
            model_type: "qwen", "llama", "mistral"
        """
        formatted = []

        for item in data:
            turns = item["turns"]

            if model_type == "qwen":
                parts = []
                for t in turns:
                    parts.append(f"<|im_start|>{t['role']}\n{t['content']}<|im_end|>")
                text = "\n".join(parts)

            elif model_type == "llama":
                parts = ["<|begin_of_text|>"]
                for t in turns:
                    parts.append(
                        f"<|start_header_id|>{t['role']}<|end_header_id|>\n\n{t['content']}<|eot_id|>"
                    )
                text = "".join(parts)

            else:  # generic
                parts = []
                for t in turns:
                    label = "User" if t["role"] == "user" else "Assistant"
                    parts.append(f"### {label}:\n{t['content']}")
                text = "\n\n".join(parts)

            formatted.append({
                "text": text,
                "turns": turns,
            })

        return formatted

    def split_data(self, data: List[Dict], train_ratio: float = 0.9, val_ratio: float = 0.05):
        """
        Train/val/test split

        Args:
            train_ratio: Training oranı
            val_ratio: Validation oranı
            test_ratio otomatik hesaplanır
        """
        random.shuffle(data)

        total = len(data)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)

        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]

        print(f"\n[OK] Split:")
        print(f"  Train: {len(train_data)}")
        print(f"  Val:   {len(val_data)}")
        print(f"  Test:  {len(test_data)}")

        return train_data, val_data, test_data

    def save_splits(
        self,
        train_data: List[Dict],
        val_data: List[Dict],
        test_data: List[Dict],
        output_dir: str = "data/splits"
    ):
        """Split'leri kaydet (JSONL format)"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # JSONL formatında kaydet (HuggingFace datasets için)
        for split_name, split_data in [
            ("train", train_data),
            ("val", val_data),
            ("test", test_data)
        ]:
            output_file = f"{output_dir}/{split_name}.jsonl"

            with jsonlines.open(output_file, 'w') as writer:
                writer.write_all(split_data)

            print(f"[OK] Kaydedildi: {output_file}")

    def process_pipeline(
        self,
        model_type: str = "qwen",
        output_dir: str = "data/splits"
    ):
        """Tam işleme pipeline'ı"""

        print("=" * 50)
        print("VERİ İŞLEME PIPELINE")
        print("=" * 50)

        # 1. Temizle
        cleaned_data = self.clean_data()

        # 2. Formatla
        print(f"\nModel formatı: {model_type}")
        formatted_data = self.format_for_training(cleaned_data, model_type)

        # 3. Split
        train, val, test = self.split_data(formatted_data)

        # 4. Kaydet
        self.save_splits(train, val, test, output_dir)

        print("\n[TAMAMLANDI] Isleme tamamlandi!")
        print(f"Toplam: {len(formatted_data)} ornek")

        return train, val, test


if __name__ == "__main__":
    # A SEÇENEĞİ: sadece yüksek kaliteli v2 verisi (eski mode-collapse verisi hariç)
    processor = DataProcessor([
        "data/raw/synthetic_v2_ckpt_1300.json",   # en zengin v2 (7792)
        "data/raw/synthetic_v2.json",             # ayrı v2 oturumu (3072)
    ])

    processor.process_pipeline(
        model_type="qwen",  # Qwen2.5-14B-Instruct (Bozdoğan) -> ChatML
        output_dir="data/splits"
    )

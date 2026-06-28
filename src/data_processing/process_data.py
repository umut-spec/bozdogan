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
    def __init__(self, input_file: str):
        self.input_file = input_file
        self.data = self._load_data()

    def _load_data(self) -> List[Dict]:
        """Ham veriyi yükle"""
        with open(self.input_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def clean_data(self) -> List[Dict]:
        """
        Veriyi temizle ve filtrele
        - Boş mesajları çıkar
        - Çok kısa/uzun mesajları filtrele
        - Duplikatları temizle
        """
        cleaned = []
        seen = set()

        print("Veri temizleniyor...")

        for item in tqdm(self.data):
            user_msg = item.get("user", "").strip()
            assistant_msg = item.get("assistant", "").strip()

            # Boş kontrolü
            if not user_msg or not assistant_msg:
                continue

            # Uzunluk kontrolü
            if len(user_msg) < 5 or len(assistant_msg) < 10:
                continue

            if len(user_msg) > 2000 or len(assistant_msg) > 4000:
                continue

            # Duplikat kontrolü
            key = f"{user_msg}|{assistant_msg}"
            if key in seen:
                continue

            seen.add(key)
            cleaned.append({
                "user": user_msg,
                "assistant": assistant_msg
            })

        print(f"[OK] Temizleme: {len(self.data)} -> {len(cleaned)} ornek")
        return cleaned

    def format_for_training(self, data: List[Dict], model_type: str = "qwen") -> List[Dict]:
        """
        Modele uygun format'a çevir

        Args:
            data: Temiz veri
            model_type: "qwen", "llama", "mistral"
        """
        formatted = []

        for item in data:
            if model_type == "qwen":
                # Qwen2.5 chat template
                text = f"<|im_start|>user\n{item['user']}<|im_end|>\n<|im_start|>assistant\n{item['assistant']}<|im_end|>"

            elif model_type == "llama":
                # Llama 3 chat template
                text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{item['user']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{item['assistant']}<|eot_id|>"

            elif model_type == "mistral":
                # Mistral chat template
                text = f"[INST] {item['user']} [/INST] {item['assistant']}"

            else:
                # Generic format
                text = f"### User:\n{item['user']}\n\n### Assistant:\n{item['assistant']}"

            formatted.append({
                "text": text,
                "user": item["user"],
                "assistant": item["assistant"]
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
    # Kullanım
    processor = DataProcessor("data/raw/synthetic_data.json")

    processor.process_pipeline(
        model_type="qwen",  # veya "llama", "mistral"
        output_dir="data/splits"
    )

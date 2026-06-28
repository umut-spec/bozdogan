"""
BOZDOĞAN KİMLİK EĞİTİMİ (Continue Training)
Mevcut Bozdoğan adapter'ı üzerine kimlik bilgisini ekler
"""

import torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
EXISTING_ADAPTER = "models/qwen-turkish-chat"   # Mevcut Bozdoğan
OUTPUT_DIR = "models/bozdogan-identity"          # Kimlik eklenmiş yeni model

print("="*60)
print("🦅 BOZDOĞAN KİMLİK EĞİTİMİ")
print("="*60)

# 1. Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# 2. Base model yükle
print("\n📦 Base model yükleniyor...")
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.bfloat16,
    trust_remote_code=True
)
base = prepare_model_for_kbit_training(base)

# 3. Mevcut Bozdoğan adapter'ı yükle (eğitilebilir olarak)
print("🦅 Mevcut Bozdoğan adapter yükleniyor...")
model = PeftModel.from_pretrained(base, EXISTING_ADAPTER, is_trainable=True)
print("✓ Bozdoğan yüklendi, kimlik eğitimine hazır")

# 4. Tokenizer
tokenizer = AutoTokenizer.from_pretrained(EXISTING_ADAPTER, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 5. Kimlik verisi yükle
print("\n📚 Kimlik eğitim verisi yükleniyor...")
dataset = load_dataset("json", data_files={
    "train": "data/splits_identity/train.jsonl",
    "validation": "data/splits_identity/val.jsonl"
})
print(f"✓ Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}")

# 6. Eğitim ayarları (DÜŞÜK LR, kısa eğitim - mevcut bilgiyi bozmasın)
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=2,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=5.0e-5,            # Düşük LR (orijinalin 1/4'ü)
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    bf16=True,
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    max_length=1024,
    packing=False,
    report_to=[],
)

# 7. Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)

print("\n🚀 Kimlik eğitimi başlıyor...\n")
trainer.train()

print("\n✅ Kimlik eğitimi tamamlandı!")
print(f"\n💾 Bozdoğan (kimlikli) kaydediliyor: {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("✓ Kaydedildi!")

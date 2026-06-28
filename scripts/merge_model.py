"""
BOZDOĞAN MERGE - Adapter'ı base model ile birleştir
Sonuç: tek parça model (serverless/vLLM için)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

BASE = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER = "UmutArchery/Bozdogan-7B"   # HuggingFace'deki adapter (Colab silinince buradan çeker)
OUTPUT = "models/bozdogan-merged"

print("🦅 MERGE BAŞLIYOR")
print("="*60)

# 1. Base model (TAM precision - merge için 4bit DEĞİL)
print("\n📦 Base model yükleniyor (fp16)...")
base = AutoModelForCausalLM.from_pretrained(
    BASE,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# 2. Adapter'ı yükle
print("🦅 Bozdoğan adapter yükleniyor...")
model = PeftModel.from_pretrained(base, ADAPTER)

# 3. MERGE
print("🔧 Birleştiriliyor...")
model = model.merge_and_unload()
print("✓ Merge tamam")

# 4. Kaydet
print(f"\n💾 Kaydediliyor: {OUTPUT}")
model.save_pretrained(OUTPUT, safe_serialization=True)

tokenizer = AutoTokenizer.from_pretrained(ADAPTER, trust_remote_code=True)
tokenizer.save_pretrained(OUTPUT)

print("\n✅ MERGE TAMAMLANDI!")
print(f"Tek parça model: {OUTPUT}")

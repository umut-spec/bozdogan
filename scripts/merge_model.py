"""
BOZDOĞAN MERGE + AYNI REPO'YA TAM MODEL YÜKLE
Adapter'ı base ile birleştirir, tek parça tam model olarak
UmutArchery/Bozdogan-7B repo'suna yükler (adapter dosyalarını siler)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import HfApi, login
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

BASE = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER = "UmutArchery/Bozdogan-7B"     # mevcut adapter repo'su
OUTPUT = "models/bozdogan-merged"
REPO = "UmutArchery/Bozdogan-7B"        # AYNI repo - tam modelle güncellenecek
HF_TOKEN = "BURAYA_HF_TOKEN"            # Write izinli token

print("🦅 MERGE + YÜKLEME BAŞLIYOR")
print("="*60)

# 1. Base model (fp16 - merge için)
print("\n📦 Base model yükleniyor...")
base = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
)

# 2. Adapter yükle
print("🦅 Bozdoğan adapter yükleniyor...")
model = PeftModel.from_pretrained(base, ADAPTER)

# 3. Merge
print("🔧 Birleştiriliyor...")
model = model.merge_and_unload()
print("✓ Merge tamam")

# 4. Yerel kaydet
print(f"\n💾 Yerel kayıt: {OUTPUT}")
model.save_pretrained(OUTPUT, safe_serialization=True)
tokenizer = AutoTokenizer.from_pretrained(ADAPTER, trust_remote_code=True)
tokenizer.save_pretrained(OUTPUT)

# 5. HuggingFace'e giriş
login(HF_TOKEN)
api = HfApi()

# README (tam model kullanımı)
readme = '''---
license: apache-2.0
language:
- tr
base_model: Qwen/Qwen2.5-7B-Instruct
tags:
- turkish
- chat
- bozdogan
pipeline_tag: text-generation
---

# 🦅 Bozdoğan-7B

Umut Archery tarafından geliştirilen Türkçe sohbet modeli. Qwen 2.5 7B üzerine ~10.000 Türkçe örnekle eğitildi. Tek parça tam modeldir (vLLM/serverless ile direkt çalışır).

## Kullanım
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("UmutArchery/Bozdogan-7B", device_map="auto", dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("UmutArchery/Bozdogan-7B")

SYSTEM = "Sen Bozdogan'sin. Umut Archery tarafindan gelistirilen bir Turkce yapay zeka asistanisin."
prompt = f"<|im_start|>system\\n{SYSTEM}<|im_end|>\\n<|im_start|>user\\nMerhaba!<|im_end|>\\n<|im_start|>assistant\\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=300, temperature=0.7, do_sample=True)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

*Bozdoğan, Umut Archery tarafından geliştirilmiştir. 🦅*
'''
with open(f"{OUTPUT}/README.md", "w", encoding="utf-8") as f:
    f.write(readme)

# 6. Eski adapter dosyalarını sil (artık tam model var)
print("\n🗑️ Eski adapter dosyaları siliniyor...")
for f in ["adapter_model.safetensors", "adapter_config.json", "README.md"]:
    try:
        api.delete_file(path_in_repo=f, repo_id=REPO)
        print(f"  silindi: {f}")
    except Exception:
        pass

# 7. Tam modeli aynı repo'ya yükle
print(f"\n📤 Tam model yükleniyor: {REPO}")
api.upload_folder(folder_path=OUTPUT, repo_id=REPO)

print("\n✅ TAMAMLANDI!")
print(f"🦅 Tek parça tam model: https://huggingface.co/{REPO}")


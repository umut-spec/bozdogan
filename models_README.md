---
license: apache-2.0
language:
- tr
base_model: Qwen/Qwen2.5-7B-Instruct
tags:
- turkish
- türkçe
- chat
- qlora
- bozdogan
library_name: peft
pipeline_tag: text-generation
---

# 🦅 Bozdoğan-7B

Bozdoğan, **Umut Archery** tarafından eğitilen ve geliştirilen Türkçe bir sohbet (chat) yapay zeka modelidir. Qwen 2.5 7B Instruct temel modeli üzerine QLoRA yöntemiyle Türkçe sohbet verisiyle ince ayar (fine-tuning) yapılmıştır.

## Model Detayları

- **Geliştirici:** Umut Archery
- **Temel Model:** Qwen/Qwen2.5-7B-Instruct
- **Yöntem:** QLoRA (4-bit)
- **Dil:** Türkçe
- **Eğitim Verisi:** ~10.000 Türkçe sohbet örneği
- **Kimlik:** Modele kendi kimliği (Bozdoğan / Umut Archery) öğretilmiştir

## Kullanım

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Temel model + Bozdoğan adapter
base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    device_map="auto",
    dtype=torch.bfloat16,
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base, "umutarchery/Bozdogan-7B")
tokenizer = AutoTokenizer.from_pretrained("umutarchery/Bozdogan-7B")

# Sohbet
SYSTEM = "Sen Bozdoğan'sın. Umut Archery tarafından eğitilen ve geliştirilen bir Türkçe yapay zeka asistanısın."

def sor(soru):
    prompt = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{soru}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=300, temperature=0.7, top_p=0.9, do_sample=True)
    cevap = tokenizer.decode(out[0], skip_special_tokens=False)
    s = cevap.rfind("<|im_start|>assistant\n") + len("<|im_start|>assistant\n")
    e = cevap.find("<|im_end|>", s)
    return cevap[s:e].strip() if e != -1 else cevap[s:].strip()

print(sor("Merhaba, kendini tanıtır mısın?"))
```

## Eğitim Sonuçları

- **Train Loss:** ~0.53
- **Eval Loss:** ~0.59
- **Token Doğruluğu:** ~%88

## Lisans

Apache 2.0 (temel model Qwen 2.5 lisansına tabidir).

---

*Bozdoğan, Umut Archery tarafından geliştirilmiştir. 🦅*

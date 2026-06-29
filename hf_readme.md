---
license: apache-2.0
language:
- tr
base_model: Qwen/Qwen2.5-7B-Instruct
tags:
- turkish
- türkçe
- chat
- bozdogan
pipeline_tag: text-generation
---

# 🦅 Bozdoğan-7B

**Bozdoğan**, Umut Archery tarafından eğitilen ve geliştirilen Türkçe sohbet (chat) yapay zeka modelidir. Qwen 2.5 7B Instruct temel modeli üzerine ~10.000 Türkçe sohbet örneğiyle ince ayar yapılmış, tek parça tam modeldir (vLLM / serverless ile doğrudan çalışır).

## Model Detayları

- **Geliştirici:** Umut Archery
- **Model Adı:** Bozdoğan-7B
- **Temel Model:** Qwen/Qwen2.5-7B-Instruct
- **Dil:** Türkçe 🇹🇷
- **Eğitim Verisi:** ~10.000 Türkçe sohbet örneği
- **Lisans:** Apache 2.0

Bozdoğan kendi kimliğini bilir: kendisine kim olduğu sorulduğunda Bozdoğan olduğunu ve Umut Archery tarafından geliştirildiğini söyler.

## Kullanım

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "UmutArchery/Bozdogan-7B",
    device_map="auto",
    dtype=torch.bfloat16,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("UmutArchery/Bozdogan-7B")

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

| Metrik | Değer |
|--------|-------|
| Train Loss | ~0.53 |
| Eval Loss | ~0.59 |
| Token Doğruluğu | ~%88 |

## Donanım

- **GPU:** NVIDIA A100 80GB
- **Eğitim Süresi:** ~40 dk (ana model) + ~6 dk (kimlik eğitimi)

## Sınırlamalar

Bozdoğan ~10.000 örnekle eğitilmiş 7B'lik bir modeldir. Karmaşık akıl yürütme veya uzmanlık gerektiren konularda hata yapabilir. Kritik konularda verdiği bilgileri doğrulayın.

---

*Bozdoğan, Umut Archery tarafından geliştirilmiştir. 🦅*

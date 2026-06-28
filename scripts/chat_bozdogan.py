"""
BOZDOĞAN SOHBET - Sistem promptu ile
Hem kimlik hem kişilik sistem promptundan gelir
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Sistem promptunu yükle
with open("configs/system_prompt.txt", 'r', encoding='utf-8') as f:
    SYSTEM_PROMPT = f.read().strip()

# Model yolu (kimlik eğitimi yaptıysan "models/bozdogan-identity", yapmadıysan "models/qwen-turkish-chat")
ADAPTER_PATH = "models/qwen-turkish-chat"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def load_bozdogan():
    print("🦅 Bozdoğan yükleniyor...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
    print("✅ Bozdoğan hazır!\n")
    return model, tokenizer


def sor(model, tokenizer, soru, max_token=300):
    """Sistem promptu ile soru sor"""
    # Qwen formatı: system + user + assistant
    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{soru}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_token,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    c = tokenizer.decode(out[0], skip_special_tokens=False)
    s = c.rfind("<|im_start|>assistant\n") + len("<|im_start|>assistant\n")
    e = c.find("<|im_end|>", s)
    return c[s:e].strip() if e != -1 else c[s:].strip()


if __name__ == "__main__":
    model, tokenizer = load_bozdogan()

    print("="*60)
    print("🦅 BOZDOĞAN - SİSTEM PROMPTU İLE TEST")
    print("="*60)

    test_sorular = [
        "Sen kimsin?",
        "Seni kim geliştirdi?",
        "Adın ne?",
        "Merhaba, nasılsın?",
        "Python'da liste nasıl oluşturulur?",
        "Akşam yemeği için ne önerirsin?",
    ]

    for soru in test_sorular:
        print(f"\n{'─'*60}")
        print(f"❓ {soru}")
        print(f"🦅 {sor(model, tokenizer, soru)}")

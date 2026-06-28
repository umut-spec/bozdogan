"""
Eğitilmiş modeli test et
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def load_model(adapter_path="models/qwen-turkish-chat", base_model="Qwen/Qwen2.5-7B-Instruct"):
    """Model ve tokenizer yükle"""

    print("🔄 Model yükleniyor...")
    print(f"  Base: {base_model}")
    print(f"  Adapter: {adapter_path}")
    print()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,
        trust_remote_code=True
    )

    # Base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    # LoRA adapter yükle
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print("✅ Model yüklendi!")
    print()

    return model, tokenizer


def chat(model, tokenizer, user_message, max_new_tokens=200):
    """Modelle sohbet et"""

    # Qwen chat template
    prompt = f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Extract assistant response
    assistant_start = response.find("<|im_start|>assistant\n") + len("<|im_start|>assistant\n")
    assistant_end = response.find("<|im_end|>", assistant_start)

    if assistant_end != -1:
        return response[assistant_start:assistant_end].strip()
    else:
        return response[assistant_start:].strip()


def run_tests(model, tokenizer):
    """Test örnekleri çalıştır"""

    test_cases = [
        "Merhaba! Türkçe model olarak bana kendini tanıtır mısın?",
        "Python'da liste ve tuple arasındaki farklar nelerdir?",
        "Bugün hava çok güzel, dışarı çıkmayı planlıyorum. Ne önerirsin?",
        "Stresli bir günden sonra rahatlamak için ne yapmalıyım?",
        "Makarna yaparken suyuna tuz ne zaman eklenir?",
    ]

    print("=" * 70)
    print("MODEL TEST SONUÇLARI")
    print("=" * 70)
    print()

    for i, user_msg in enumerate(test_cases, 1):
        print(f"Test #{i}")
        print(f"{'─' * 70}")
        print(f"Kullanıcı: {user_msg}")
        print()

        response = chat(model, tokenizer, user_msg)

        print(f"Model: {response}")
        print()
        print(f"{'=' * 70}")
        print()


def interactive_mode(model, tokenizer):
    """İnteraktif sohbet modu"""

    print("=" * 70)
    print("İNTERAKTİF SOHBET MODU")
    print("=" * 70)
    print()
    print("Çıkmak için 'exit' veya 'quit' yazın")
    print()

    while True:
        user_input = input("Siz: ").strip()

        if user_input.lower() in ['exit', 'quit', 'çık']:
            print("\nGörüşmek üzere! 👋")
            break

        if not user_input:
            continue

        response = chat(model, tokenizer, user_input)
        print(f"Model: {response}")
        print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tuned Qwen modelini test et")
    parser.add_argument("--adapter", type=str, default="models/qwen-turkish-chat",
                        help="LoRA adapter yolu")
    parser.add_argument("--base", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Base model adı")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="İnteraktif sohbet modu")

    args = parser.parse_args()

    # Model yükle
    model, tokenizer = load_model(args.adapter, args.base)

    if args.interactive:
        # İnteraktif mod
        interactive_mode(model, tokenizer)
    else:
        # Test örnekleri
        run_tests(model, tokenizer)

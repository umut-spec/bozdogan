"""
Fine-tune edilmiş modeli test et
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse


def load_finetuned_model(model_path: str, base_model: str = "Qwen/Qwen2.5-7B-Instruct"):
    """Fine-tune edilmiş modeli yükle"""

    print(f"Base model yükleniyor: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    # LoRA weights yükle
    print(f"LoRA weights yükleniyor: {model_path}")
    model = PeftModel.from_pretrained(base_model, model_path)
    model = model.merge_and_unload()  # LoRA'yı base model'e merge et

    model.eval()

    print("✓ Model yüklendi")
    return model, tokenizer


def chat(model, tokenizer, user_input: str, max_new_tokens: int = 512):
    """Model ile konuşma"""

    # Qwen2.5 chat template
    messages = [
        {"role": "user", "content": user_input}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response


def interactive_test(model_path: str):
    """İnteraktif test modu"""

    model, tokenizer = load_finetuned_model(model_path)

    print("\n" + "=" * 60)
    print("TÜRKÇE CHAT MODEL TESTİ")
    print("=" * 60)
    print("Çıkmak için 'quit' yazın\n")

    while True:
        user_input = input("Siz: ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Görüşmek üzere!")
            break

        if not user_input:
            continue

        print("\nModel: ", end="", flush=True)
        response = chat(model, tokenizer, user_input)
        print(response)
        print()


def batch_test(model_path: str, test_prompts: list):
    """Batch test - örnek promptlarla"""

    model, tokenizer = load_finetuned_model(model_path)

    print("\n" + "=" * 60)
    print("BATCH TEST")
    print("=" * 60)

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[{i}/{len(test_prompts)}] Prompt: {prompt}")
        response = chat(model, tokenizer, prompt)
        print(f"Response: {response}")
        print("-" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Fine-tuned model path")
    parser.add_argument("--mode", type=str, default="interactive", choices=["interactive", "batch"])
    args = parser.parse_args()

    if args.mode == "interactive":
        interactive_test(args.model_path)

    else:
        # Örnek test promptları
        test_prompts = [
            "Merhaba, nasılsın?",
            "Python'da liste ve tuple arasındaki fark nedir?",
            "İstanbul'da gezilecek yerler önerir misin?",
            "Sağlıklı bir kahvaltı için ne önerirsin?",
            "Yapay zeka nedir ve nasıl çalışır?"
        ]

        batch_test(args.model_path, test_prompts)

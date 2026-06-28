"""
Google Colab için tek komutla kurulum ve hazırlık
Colab notebook'ta bu scripti çalıştırın
"""

import subprocess
import sys
import os
from pathlib import Path

print("="*70)
print("🚀 GOOGLE COLAB KURULUM")
print("="*70)
print()

# 1. GPU Kontrolü
print("1️⃣ GPU Kontrolü...")
result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                       capture_output=True, text=True)
if result.returncode == 0:
    gpu_info = result.stdout.strip()
    print(f"✅ GPU Bulundu: {gpu_info}")
else:
    print("❌ GPU bulunamadı! Runtime > Change runtime type > T4 GPU seçin")
    sys.exit(1)

print()

# 2. Paket Kurulumu
print("2️⃣ Paketler kuruluyor...")
packages = [
    "transformers",
    "peft",
    "accelerate",
    "bitsandbytes",
    "trl",
    "datasets",
    "jsonlines",
    "pyyaml",
    "rich"
]

for package in packages:
    print(f"  Installing {package}...", end=" ")
    result = subprocess.run([sys.executable, "-m", "pip", "install", "-q", package],
                          capture_output=True)
    if result.returncode == 0:
        print("✅")
    else:
        print(f"❌ {result.stderr.decode()}")

print()

# 3. PyTorch kontrol
print("3️⃣ PyTorch CUDA kontrolü...")
try:
    import torch
    if torch.cuda.is_available():
        print(f"✅ PyTorch CUDA: {torch.version.cuda}")
        print(f"✅ Device: {torch.cuda.get_device_name(0)}")
        print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠️  CUDA mevcut değil!")
except ImportError:
    print("❌ PyTorch kurulu değil!")

print()

# 4. Veri kontrolü
print("4️⃣ Veri dosyaları kontrol ediliyor...")
data_files = [
    "data/splits/train.jsonl",
    "data/splits/val.jsonl",
    "data/splits/test.jsonl"
]

all_exist = True
for file_path in data_files:
    if Path(file_path).exists():
        size = Path(file_path).stat().st_size / 1024  # KB
        print(f"✅ {file_path} ({size:.1f} KB)")
    else:
        print(f"❌ {file_path} bulunamadı!")
        all_exist = False

if not all_exist:
    print()
    print("⚠️  Veri dosyaları eksik!")
    print("Çözüm: Repo'yu klonladıktan sonra bu scripti çalıştırın")

print()

# 5. Config dosyası oluştur
print("5️⃣ Colab config oluşturuluyor...")
config_template = """# Model Konfigürasyonu
model_name: "Qwen/Qwen2.5-7B-Instruct"
output_dir: "/content/qwen-turkish-chat"

# LoRA Parametreleri
lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
  bias: "none"
  task_type: "CAUSAL_LM"

# QLoRA için Quantization
quantization:
  load_in_4bit: true
  bnb_4bit_compute_dtype: "bfloat16"
  bnb_4bit_use_double_quant: true
  bnb_4bit_quant_type: "nf4"

# Training Parametreleri - COLAB OPTIMIZED
training:
  num_train_epochs: 3
  per_device_train_batch_size: 2      # T4 için düşük
  per_device_eval_batch_size: 2
  gradient_accumulation_steps: 8      # Effective batch = 16
  learning_rate: 2.0e-4
  lr_scheduler_type: "cosine"
  warmup_ratio: 0.03
  weight_decay: 0.01
  max_grad_norm: 1.0

  # Optimization
  optim: "paged_adamw_8bit"
  gradient_checkpointing: true
  fp16: false
  bf16: true

  # Logging & Saving
  logging_steps: 20
  save_strategy: "steps"
  save_steps: 200                     # Daha seyrek checkpoint
  save_total_limit: 2                 # Disk tasarrufu
  evaluation_strategy: "steps"
  eval_steps: 200

  # Dataset
  max_seq_length: 2048

  # Distributed
  ddp_find_unused_parameters: false

# Veri Dosyaları
data:
  train_file: "data/splits/train.jsonl"
  val_file: "data/splits/val.jsonl"
  test_file: "data/splits/test.jsonl"
"""

config_path = Path("configs/colab_config.yaml")
config_path.parent.mkdir(parents=True, exist_ok=True)
config_path.write_text(config_template, encoding='utf-8')
print(f"✅ Config oluşturuldu: {config_path}")

print()

# 6. Özet
print("="*70)
print("📋 KURULUM TAMAMLANDI!")
print("="*70)
print()
print("Eğitimi başlatmak için:")
print()
print("  !python src/training/train.py --config configs/colab_config.yaml")
print()
print("TensorBoard ile izlemek için:")
print()
print("  %load_ext tensorboard")
print("  %tensorboard --logdir /content/qwen-turkish-chat")
print()
print("💡 İpucu: Model'i Drive'a kaydetmek için output_dir'i değiştirin:")
print("  output_dir: '/content/drive/MyDrive/qwen-turkish-chat'")
print()

# 🚀 Google Colab'da Eğitim Rehberi

## Hızlı Başlangıç (5 Dakika)

### 1. Colab'ı Aç
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/turkish-llm-finetuning/blob/main/notebooks/colab_training.ipynb)

### 2. GPU Seç
- **Runtime** > **Change runtime type** > **T4 GPU** (veya daha iyisi)

### 3. Hızlı Kurulum (Tek Hücre)

```python
# Repo'yu klonla
!git clone https://github.com/YOUR_USERNAME/turkish-llm-finetuning.git
%cd turkish-llm-finetuning

# Kurulum scriptini çalıştır
!python scripts/colab_setup.py

# Eğitimi başlat
!python src/training/train.py --config configs/colab_config.yaml
```

## GPU Seçenekleri

| GPU | VRAM | Eğitim Süresi | Batch Size | Colab Plan |
|-----|------|---------------|------------|------------|
| T4 | 16GB | 2-3 saat | 2 | Free/Pro |
| L4 | 24GB | 1-1.5 saat | 4 | Pro+ |
| A100 | 40GB | 30-45 dakika | 8 | Pro+ |

## Detaylı Adımlar

### 1. Repo Klonlama

```python
!git clone https://github.com/YOUR_USERNAME/turkish-llm-finetuning.git
%cd turkish-llm-finetuning
!ls -la
```

### 2. Bağımlılıkları Kurma

```python
!pip install -q transformers peft accelerate bitsandbytes trl datasets
!pip install -q jsonlines pyyaml rich
```

### 3. GPU Kontrolü

```python
!nvidia-smi
```

```python
import torch
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
```

### 4. Veriyi Doğrulama

```python
!python scripts/validate_data.py
```

### 5. Config Ayarlama (GPU'ya Göre)

#### T4 GPU için:
```python
import yaml

with open('configs/training_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

config['training']['per_device_train_batch_size'] = 2
config['training']['gradient_accumulation_steps'] = 8
config['training']['save_steps'] = 200
config['training']['eval_steps'] = 200

with open('configs/colab_config.yaml', 'w') as f:
    yaml.dump(config, f)

print("✅ T4 config hazır")
```

#### A100 GPU için:
```python
config['training']['per_device_train_batch_size'] = 8
config['training']['gradient_accumulation_steps'] = 2
config['training']['save_steps'] = 100
config['training']['eval_steps'] = 100

with open('configs/colab_config.yaml', 'w') as f:
    yaml.dump(config, f)

print("✅ A100 config hazır")
```

### 6. Google Drive Bağlama (Önerilen)

Model ve checkpointleri Drive'a kaydetmek için:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Config'de output_dir'i güncelle:
```python
config['output_dir'] = '/content/drive/MyDrive/qwen-turkish-chat'
```

### 7. Eğitimi Başlatma

```python
!python src/training/train.py --config configs/colab_config.yaml
```

### 8. TensorBoard ile İzleme

Ayrı bir hücrede:
```python
%load_ext tensorboard
%tensorboard --logdir /content/qwen-turkish-chat
```

### 9. Model Testi

Eğitim tamamlandıktan sonra:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Model yükle
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

model = PeftModel.from_pretrained(base_model, "/content/qwen-turkish-chat")
tokenizer = AutoTokenizer.from_pretrained("/content/qwen-turkish-chat", trust_remote_code=True)

# Test fonksiyonu
def chat(user_message):
    prompt = f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    start = response.find("<|im_start|>assistant\n") + len("<|im_start|>assistant\n")
    end = response.find("<|im_end|>", start)
    return response[start:end].strip()

# Test
print(chat("Merhaba! Nasılsın?"))
```

### 10. Model'i Kaydetme

#### Drive'a Kaydetme
```python
# Zaten Drive'a bağlıysanız otomatik kaydedilir
!ls -lh /content/drive/MyDrive/qwen-turkish-chat
```

#### HuggingFace Hub'a Yükleme
```python
from huggingface_hub import notebook_login

notebook_login()

model.push_to_hub("YOUR_USERNAME/qwen-2.5-7b-turkish-chat")
tokenizer.push_to_hub("YOUR_USERNAME/qwen-2.5-7b-turkish-chat")
```

#### ZIP'leyip İndirme
```python
!cd /content && zip -r qwen-turkish-chat.zip qwen-turkish-chat

from google.colab import files
files.download('/content/qwen-turkish-chat.zip')
```

## 🐛 Sorun Giderme

### Out of Memory (OOM)

```python
# Batch size'ı azalt
config['training']['per_device_train_batch_size'] = 1
config['training']['gradient_accumulation_steps'] = 16

# Sequence length'i azalt
config['training']['max_seq_length'] = 1024
```

### Colab Disconnect

```python
# Periyodik olarak checkpoint kaydet
config['training']['save_steps'] = 100

# Drive'a kaydet
config['output_dir'] = '/content/drive/MyDrive/qwen-turkish-chat'
```

Resume etmek için:
```python
!python src/training/train.py \
    --config configs/colab_config.yaml \
    --resume_from_checkpoint /content/drive/MyDrive/qwen-turkish-chat/checkpoint-XXX
```

### Yavaş Eğitim

```python
# Gradient checkpointing'i kapat (daha fazla VRAM kullanır)
config['training']['gradient_checkpointing'] = False

# Batch size'ı artır (A100 için)
config['training']['per_device_train_batch_size'] = 8
```

## 💡 Pro İpuçları

### 1. Background Tab Kapatma
Eğitim sırasında sayfayı kapatabilirsiniz. Colab çalışmaya devam eder.

### 2. Checkpoint Stratejisi
```python
# Her 200 step'te kaydet
save_steps: 200
save_total_limit: 2  # Sadece son 2 checkpoint'i tut (disk tasarrufu)
```

### 3. Evaluation Stratejisi
```python
# Daha sık evaluation
eval_steps: 100

# Daha seyrek (hız için)
eval_steps: 500
```

### 4. Logging
```python
# Detaylı logging
logging_steps: 10

# Az logging (hız için)
logging_steps: 50
```

### 5. Multi-GPU (A100 x 2)
```python
# Colab Pro+'da 2x A100 varsa
!python src/training/train.py --config configs/colab_config.yaml --nproc_per_node 2
```

## 📊 Beklenen Sonuçlar

### Eğitim Metrikleri
- **Başlangıç Loss:** ~2.5-3.0
- **Final Loss:** ~0.5-1.0
- **Eval Loss:** ~0.8-1.2

### Süre (10k örnekle)
- **T4:** 2-3 saat
- **L4:** 1-1.5 saat
- **A100:** 30-45 dakika

## 🎯 Sonraki Adımlar

1. ✅ Model eğitimi tamamlandı
2. 📊 Test set üzerinde değerlendir
3. 🧪 Çeşitli promptlarla test et
4. 🚀 HuggingFace Hub'a yükle
5. 📱 Gradio demo oluştur

## 📝 Notlar

- Colab Free: 12 saat session limit
- Colab Pro: 24 saat session limit
- GPU bazen değişebilir (T4 → L4), eğitim durunca resume edin
- Model ~6GB (adapter weights)
- Checkpoint'ler ~2GB her biri

## 🔗 Kaynaklar

- [Colab Training Notebook](notebooks/colab_training.ipynb)
- [Training Guide](TRAINING_GUIDE.md)
- [Qwen 2.5 Docs](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [PEFT/LoRA Docs](https://huggingface.co/docs/peft)

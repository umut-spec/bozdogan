# Eğitim Rehberi

## ✅ Hazırlık Durumu

- ✅ Veri işlendi: **10,591 örnek**
- ✅ Train/Val/Test split: **9,531 / 529 / 531**
- ✅ Format: **Qwen2.5 chat template**
- ✅ Kalite kontrolü: **Geçti**

## 🚀 Hızlı Başlangıç

### 1. Bağımlılıkları Kur

```bash
pip install -r requirements.txt
```

### 2. Veriyi Doğrula (Opsiyonel)

```bash
python scripts/validate_data.py
```

### 3. Eğitime Başla

#### Yerel (GPU ile)
```bash
python src/training/train.py
```

#### Vast.ai/Cloud (önerilen)
```bash
# SSH ile bağlan
ssh -p <PORT> root@<IP>

# Repo'yu klonla
git clone <YOUR_REPO_URL>
cd turkish-llm-finetuning

# Bağımlılıkları kur
pip install -r requirements.txt

# Eğitimi başlat
python src/training/train.py --config configs/training_config.yaml
```

## 📊 Eğitim Parametreleri

### Model
- **Base:** Qwen/Qwen2.5-7B-Instruct
- **Method:** QLoRA (4-bit)
- **LoRA Rank:** 16
- **Target Modules:** q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

### Hiperparametreler
- **Epochs:** 3
- **Batch Size:** 4 (per device)
- **Gradient Accumulation:** 4 (effective batch = 16)
- **Learning Rate:** 2e-4
- **LR Scheduler:** Cosine with warmup
- **Optimizer:** paged_adamw_8bit
- **Max Sequence Length:** 2048

### Donanım Gereksinimleri
- **Minimum:** 24GB VRAM (RTX 3090, 4090)
- **Önerilen:** 48GB+ VRAM (H100, H200)
- **Tahmini Süre:** 1-2 saat (H200 NVL)

## 📁 Çıktılar

Eğitim sonunda şunlar oluşur:

```
models/qwen-turkish-chat/
├── adapter_config.json          # LoRA konfigürasyonu
├── adapter_model.safetensors    # LoRA weights
├── tokenizer_config.json
├── tokenizer.json
└── special_tokens_map.json
```

## 🔍 İzleme

### TensorBoard
```bash
tensorboard --logdir models/qwen-turkish-chat
```

### Training Logs
Eğitim sırasında:
- Her 10 step'te log
- Her 100 step'te evaluation
- Her 100 step'te checkpoint save

## 🧪 Model Testi

Eğitim bittikten sonra:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Base model yükle
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    device_map="auto"
)

# LoRA adapter'ı yükle
model = PeftModel.from_pretrained(base_model, "models/qwen-turkish-chat")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("models/qwen-turkish-chat")

# Test
prompt = "<|im_start|>user\nMerhaba, nasılsın?<|im_end|>\n<|im_start|>assistant\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

## 💰 Maliyet Tahmini

### Vast.ai (H200 NVL)
- **Ücret:** ~$3-4/saat
- **Eğitim:** 1-2 saat
- **Toplam:** ~$5-8

### Cloud GPU (A100/H100)
- **RunPod:** ~$2-3/saat
- **Lambda Labs:** ~$1.5-2.5/saat

## 🎯 Sonraki Adımlar

1. ✅ Model eğitimi tamamlandı mı?
2. 📊 Evaluation metrikleri kontrol et
3. 🧪 Test set üzerinde performans ölç
4. 🚀 Model'i deploy et (HuggingFace Hub, vLLM, etc.)
5. 📈 Daha fazla veri ile iteration yap

## ⚙️ Konfigürasyon Özelleştirme

`configs/training_config.yaml` dosyasını düzenle:

```yaml
# Daha hızlı eğitim için
training:
  num_train_epochs: 2
  per_device_train_batch_size: 8  # Daha fazla VRAM gereklı

# Daha iyi kalite için
lora:
  r: 32  # Daha yüksek rank
  lora_alpha: 64

# Daha uzun context için
training:
  max_seq_length: 4096  # Daha fazla VRAM gerekli
```

## 🐛 Sorun Giderme

### Out of Memory (OOM)
```yaml
# Batch size'ı azalt
per_device_train_batch_size: 2
gradient_accumulation_steps: 8

# Sequence length'i azalt
max_seq_length: 1024
```

### Slow Training
```yaml
# Gradient checkpointing'i kapat (daha fazla VRAM kullanır)
gradient_checkpointing: false

# Batch size'ı artır
per_device_train_batch_size: 8
```

### Poor Performance
- Daha fazla epoch dene (3-5)
- Learning rate'i ayarla (1e-4 - 5e-4)
- Daha fazla data ekle

## 📞 Destek

Sorularınız için:
- Issues: GitHub Issues
- Documentation: `docs/` klasörü
- Examples: `notebooks/` klasörü

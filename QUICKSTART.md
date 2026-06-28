# Turkish LLM Fine-tuning - Quick Start

## 📋 Gereksinimler

- Python 3.9+
- CiciAPI Key
- Vast.ai hesabı (training için)
- ~10 yuan (~$1.4) - veri üretimi
- ~$6-8 - GPU kiralama

## 🚀 Hızlı Başlangıç

### 1. Kurulum

```bash
cd turkish-llm-finetuning

# Virtual environment (önerilen)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Dependencies
pip install -r requirements.txt
```

### 2. API Key Ayarlama

```bash
# .env dosyası oluştur
cp .env.example .env

# .env'i düzenle ve CICI_API_KEY ekle
```

### 3. Sentetik Veri Üretimi

```bash
# 1000 istek × 20 konuşma = 20,000 örnek
# Maliyet: 10 yuan (~$1.4)
# Süre: ~15-20 dakika

python src/data_generation/generate_synthetic_data.py
```

**Not:** Checkpoint'ler her 100 istekte kaydedilir. İstek kesilirse kaldığınız yerden devam edebilirsiniz.

### 4. Veri İşleme

```bash
# Veriyi temizle ve formatla
python src/data_processing/process_data.py
```

Çıktı:
- `data/splits/train.jsonl` - Training data
- `data/splits/val.jsonl` - Validation data
- `data/splits/test.jsonl` - Test data

### 5. Fine-tuning (Vast.ai'de)

#### Vast.ai Setup

1. **GPU Kirala:**
   - H200 NVL (önerilen) veya H100/A100 80GB
   - Min 80GB VRAM, 100GB disk

2. **SSH Bağlan:**
   ```bash
   ssh -p <PORT> root@<IP>
   ```

3. **Projeyi Yükle:**
   ```bash
   # Vast.ai instance'ında
   cd /workspace
   git clone <YOUR_REPO> turkish-llm-finetuning
   cd turkish-llm-finetuning
   pip install -r requirements.txt
   ```

4. **Veri Dosyalarını Kopyala:**
   ```bash
   # Local'den Vast.ai'e
   scp -P <PORT> -r data/splits root@<IP>:/workspace/turkish-llm-finetuning/data/
   ```

#### Training Başlat

```bash
# Vast.ai'de
cd /workspace/turkish-llm-finetuning

# Screen ile arka planda
screen -S training
python src/training/train.py --config configs/training_config.yaml

# Ctrl+A, D ile detach
# screen -r training ile geri dön
```

**Tahmini Süre:** 1-2 saat (20K örnek, 3 epoch)

#### TensorBoard İzleme

```bash
# Vast.ai'de başka bir terminal
tensorboard --logdir=models/qwen-turkish-chat --port=6006

# Local'den port forward
ssh -L 6006:localhost:6006 -p <PORT> root@<IP>

# Tarayıcıda: http://localhost:6006
```

### 6. Model İndirme

Training bitince:

```bash
# Local bilgisayara
scp -P <PORT> -r root@<IP>:/workspace/turkish-llm-finetuning/models/qwen-turkish-chat ./models/
```

### 7. Test Etme

```bash
# İnteraktif test
python src/evaluation/test_model.py \
  --model_path models/qwen-turkish-chat \
  --mode interactive

# Batch test
python src/evaluation/test_model.py \
  --model_path models/qwen-turkish-chat \
  --mode batch
```

## 📊 Beklenen Sonuçlar

- **Veri:** 18,000-20,000 temiz konuşma örneği
- **Training süresi:** 1-2 saat
- **Model boyutu:** ~100MB (LoRA weights)
- **Inference:** CPU'da bile hızlı

## 💰 Toplam Maliyet

| İşlem | Maliyet |
|-------|---------|
| Veri üretimi (1000 istek) | $1.4 |
| GPU kiralama (1.5 saat) | $5-6 |
| **TOPLAM** | **~$7** |

## 🔧 Yapılandırma Özelleştirme

### Model Değiştirme

`configs/training_config.yaml`:
```yaml
# Qwen 2.5 7B (önerilen - Türkçe çok iyi)
model_name: "Qwen/Qwen2.5-7B-Instruct"

# veya Llama 3.2 3B (daha küçük, hızlı)
# model_name: "meta-llama/Llama-3.2-3B-Instruct"

# veya Mistral 7B
# model_name: "mistralai/Mistral-7B-Instruct-v0.3"
```

### Veri Miktarı

`src/data_generation/generate_synthetic_data.py`:
```python
generator.generate_dataset(
    num_requests=1000,           # İstek sayısı
    conversations_per_request=20, # İstek başına örnek
    # Toplam: 1000 × 20 = 20,000 örnek
)
```

### Training Parametreleri

`configs/training_config.yaml`:
```yaml
training:
  num_train_epochs: 3          # Epoch sayısı
  per_device_train_batch_size: 4  # Batch size (OOM olursa 2'ye düşür)
  learning_rate: 2.0e-4        # Learning rate
  max_seq_length: 2048         # Max token uzunluğu
```

## 🐛 Sorun Giderme

### OOM (Out of Memory)
```yaml
# training_config.yaml'da:
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
```

### Yavaş Veri Üretimi
```python
# generate_synthetic_data.py'de:
time.sleep(0.05)  # 0.1'den 0.05'e düşür
```

### CUDA Hatası
```bash
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## 📚 Daha Fazla Bilgi

- [Vast.ai Setup Guide](docs/VAST_AI_SETUP.md)
- [Training Config Açıklamaları](configs/training_config.yaml)
- [Veri Formatı](docs/DATA_FORMAT.md)

## 🎯 Sonraki Adımlar

1. **Veri kalitesini artırma:** Daha çeşitli promptlar
2. **Domain-specific fine-tuning:** Özel alanlar için veri
3. **Quantization:** GGUF/AWQ ile model küçültme
4. **Deployment:** vLLM/TGI ile serving

## 📝 Notlar

- İlk defa yapıyorsanız **küçük bir test yapın:**
  - 50 istek × 10 örnek = 500 örnek
  - 1 epoch training
  - Toplam: <30 dakika, <$1

- **Checkpoint'leri saklayın:** Training kesintiye uğrarsa devam edebilirsiniz

- **Vast.ai instance'ı unutmayın:** Training bitince MUTLAKA sonlandırın!

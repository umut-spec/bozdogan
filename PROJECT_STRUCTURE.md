# Proje Yapısı

```
turkish-llm-finetuning/
│
├── README.md                      # Genel açıklama
├── QUICKSTART.md                  # Hızlı başlangıç rehberi
├── requirements.txt               # Python bağımlılıkları
├── .env.example                   # API key template
├── .gitignore                     # Git ignore kuralları
│
├── configs/
│   └── training_config.yaml       # Training parametreleri (LoRA, QLoRA, vb.)
│
├── docs/
│   └── VAST_AI_SETUP.md          # Vast.ai kurulum ve kullanım rehberi
│
├── data/
│   ├── raw/                       # Ham sentetik veri (JSON)
│   ├── processed/                 # İşlenmiş veri
│   └── splits/                    # train/val/test splits (JSONL)
│
├── src/
│   ├── data_generation/
│   │   └── generate_synthetic_data.py   # Gemini ile veri üretimi
│   │
│   ├── data_processing/
│   │   └── process_data.py              # Veri temizleme ve formatlama
│   │
│   ├── training/
│   │   └── train.py                     # QLoRA fine-tuning scripti
│   │
│   └── evaluation/
│       └── test_model.py                # Model test ve değerlendirme
│
├── scripts/                       # Utility scriptleri
│
├── notebooks/                     # Jupyter notebooklar
│
└── models/                        # Fine-tuned model weights (gitignore'da)
    └── qwen-turkish-chat/         # Training çıktısı
```

## Ana Dosyalar

### 1. `src/data_generation/generate_synthetic_data.py`
- Gemini 3 Flash API kullanarak sentetik Türkçe konuşma verisi üretir
- CiciAPI + OpenAI SDK
- 1000 istek × 20 konuşma = 20,000 örnek
- Checkpoint sistemi (her 100 istek)
- Maliyet: ~10 yuan ($1.4)

### 2. `src/data_processing/process_data.py`
- Ham veriyi temizler ve filtreler
- Model formatına çevirir (Qwen/Llama/Mistral)
- Train/val/test split yapar (90/5/5)
- JSONL formatında kaydeder

### 3. `src/training/train.py`
- QLoRA ile fine-tuning
- Qwen 2.5 7B base model
- H200 NVL için optimize
- 1-2 saat training süresi
- TensorBoard logging

### 4. `src/evaluation/test_model.py`
- Fine-tuned modeli test eder
- İnteraktif chat modu
- Batch test modu

### 5. `configs/training_config.yaml`
- Model: Qwen 2.5 7B
- LoRA rank: 16
- Batch size: 4, gradient accumulation: 4
- Learning rate: 2e-4
- Epochs: 3
- BFloat16 precision

## Kullanım Akışı

```bash
# 1. Kurulum
pip install -r requirements.txt

# 2. API key ayarla
cp .env.example .env
# .env'e CICI_API_KEY ekle

# 3. Veri üret (Local'de)
python src/data_generation/generate_synthetic_data.py

# 4. Veri işle (Local'de)
python src/data_processing/process_data.py

# 5. Training (Vast.ai'de)
python src/training/train.py --config configs/training_config.yaml

# 6. Test et (Local veya Vast.ai'de)
python src/evaluation/test_model.py --model_path models/qwen-turkish-chat
```

## Toplam Maliyet

- Veri üretimi: $1.4
- GPU kiralama: $5-6
- **TOPLAM: ~$7**

## Tahmini Süreler

- Veri üretimi: 15-20 dakika
- Veri işleme: <5 dakika
- Training: 1-2 saat
- **TOPLAM: ~2-2.5 saat**

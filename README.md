# 🇹🇷 Turkish LLM Fine-tuning Project

Gemini 3 Flash ile sentetik veri üretimi ve Türkçe chat modeli fine-tuning projesi.

**✅ Durum:** Eğitime hazır! 10,591 temiz örnek ile Qwen 2.5 7B modelini fine-tune edebilirsiniz.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/colab_training.ipynb)

## 📁 Proje Yapısı

```
turkish-llm-finetuning/
├── data/
│   └── splits/              # ✅ Train/val/test splits (READY!)
│       ├── train.jsonl      # 9,531 örnek
│       ├── val.jsonl        # 529 örnek
│       └── test.jsonl       # 531 örnek
├── src/
│   ├── data_generation/     # Sentetik veri üretimi
│   ├── data_processing/     # Veri temizleme ve formatlama
│   ├── training/            # Fine-tuning kodu
│   └── evaluation/          # Model değerlendirme
├── configs/
│   ├── training_config.yaml     # Training ayarları
│   └── system_prompt.txt        # Bozdoğan sistem prompt'u
├── scripts/
│   ├── process_raw_data.py      # Ham veriyi işle
│   ├── validate_data.py         # Veri kalite kontrolü
│   ├── check_training_ready.py  # Sistem hazırlık kontrolü
│   ├── train_identity.py        # Kimlik eğitimi
│   ├── merge_model.py           # Adapter + base merge
│   ├── chat_bozdogan.py         # Model ile sohbet
│   └── colab_setup.py           # Colab hızlı kurulum
├── docs/                    # Kurulum rehberleri (Vast.ai vb.)
├── TRAINING_GUIDE.md        # Detaylı eğitim rehberi
├── COLAB_GUIDE.md           # Google Colab rehberi
└── README.md
```

## 🚀 Hızlı Başlangıç

### Google Colab'da Eğitim (Önerilen)

1. **Colab notebook'u aç:** [colab_training.ipynb](notebooks/colab_training.ipynb)
2. **GPU seç:** Runtime > Change runtime type > T4 GPU
3. **Tek komutla başla:**
```python
!git clone YOUR_REPO_URL
%cd turkish-llm-finetuning
!python scripts/colab_setup.py
!python src/training/train.py --config configs/training_config.yaml
```

Detaylı rehber: [COLAB_GUIDE.md](COLAB_GUIDE.md)

### Yerel veya Cloud GPU

1. **Bağımlılıkları kur:**
```bash
pip install -r requirements.txt
```

2. **Veriyi kontrol et:**
```bash
python scripts/validate_data.py
```

3. **Eğitimi başlat:**
```bash
python src/training/train.py
```

Detaylı rehber: [TRAINING_GUIDE.md](TRAINING_GUIDE.md)

## 📊 Veri İstatistikleri

- **Toplam:** 10,591 benzersiz örnek
- **Train:** 9,531 örnek (90%)
- **Validation:** 529 örnek (5%)
- **Test:** 531 örnek (5%)
- **Format:** Qwen 2.5 chat template
- **Kalite:** ✅ Doğrulandı

## 🎯 Model Bilgileri

- **Base Model:** Qwen/Qwen2.5-7B-Instruct
- **Method:** QLoRA (4-bit quantization)
- **LoRA Rank:** 16
- **Dataset Size:** 10,591 örnekler
- **Training Time:** 
  - T4 GPU: 2-3 saat
  - A100 GPU: 30-45 dakika
- **VRAM:** ~16GB (T4'te çalışır!)

## 💰 Maliyet

### Veri Üretimi (Tamamlandı ✅)
- ~10 yuan (1000+ istek × 0.010 yuan)

### GPU Eğitim
- **Google Colab Free:** $0 (T4, 12 saat limit)
- **Colab Pro:** $10/ay (T4/L4/A100, 24 saat limit)
- **Vast.ai H200:** ~$3-4/saat
- **RunPod A100:** ~$2-3/saat

**Önerilen:** Google Colab Pro (~$10/ay, sınırsız eğitim)

# Vast.ai Setup Guide

Vast.ai'de H200 NVL GPU kiralama ve fine-tuning rehberi.

## 1. Vast.ai'de GPU Kiralama

### GPU Seçimi
- **GPU:** NVIDIA H200 NVL (veya H100, A100 80GB)
- **VRAM:** Min 80GB
- **Disk:** Min 100GB
- **Bandwidth:** Yüksek (veri indirme için)

### Filtreler
```
- GPU: H200 NVL
- VRAM: 80+ GB
- DLPerf: 100+
- Reliability: 99%+
- Disk Space: 100+ GB
```

### Tahmini Maliyet
- H200 NVL: ~$3-4/saat
- Training süresi: 1-2 saat
- **Toplam:** ~$6-8

## 2. Instance Kurulumu

### Template Seçimi
- **Image:** `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel`
- veya: `nvidia/cuda:12.1.0-devel-ubuntu22.04`

### SSH ile Bağlanma
```bash
ssh -p <PORT> root@<IP_ADDRESS>
```

## 3. Ortam Kurulumu

```bash
# 1. NVIDIA drivers kontrol
nvidia-smi

# 2. Python ve pip güncellemeleri
apt-get update
apt-get install -y python3-pip git

# 3. Projeyi klonla/yükle
cd /workspace
# GitHub'dan klonla veya dosyaları kopyala
git clone <YOUR_REPO_URL>
cd turkish-llm-finetuning

# veya SCP ile kopyala (local'den):
# scp -P <PORT> -r turkish-llm-finetuning root@<IP>:/workspace/

# 4. Dependencies yükle
pip install -r requirements.txt

# 5. Flash Attention (opsiyonel, hızlandırır)
pip install flash-attn --no-build-isolation

# 6. Veri dosyalarını kontrol et
ls -lh data/splits/
```

## 4. Training Başlatma

```bash
# Tek GPU
python src/training/train.py --config configs/training_config.yaml

# Multi-GPU (eğer birden fazla GPU varsa)
torchrun --nproc_per_node=2 src/training/train.py --config configs/training_config.yaml
```

### Screen/Tmux ile Arka Planda Çalıştırma

SSH bağlantısı kopsa bile training devam etsin:

```bash
# Screen ile
screen -S training
python src/training/train.py --config configs/training_config.yaml
# Ctrl+A, D ile detach
# screen -r training ile geri dön

# veya Tmux ile
tmux new -s training
python src/training/train.py --config configs/training_config.yaml
# Ctrl+B, D ile detach
# tmux attach -t training ile geri dön
```

## 5. Training İzleme

### TensorBoard
```bash
# Başka bir terminal'de
tensorboard --logdir=models/qwen-turkish-chat --port=6006

# Local'den erişim için port forward:
# ssh -L 6006:localhost:6006 -p <PORT> root@<IP>
# Tarayıcıda: http://localhost:6006
```

### GPU Kullanımı İzleme
```bash
watch -n 1 nvidia-smi
```

### Logları İzleme
```bash
tail -f models/qwen-turkish-chat/trainer_log.txt
```

## 6. Training Tamamlandıktan Sonra

### Model'i İndirme
```bash
# Local bilgisayara SCP ile
scp -P <PORT> -r root@<IP>:/workspace/turkish-llm-finetuning/models/qwen-turkish-chat ./
```

### veya HuggingFace Hub'a Yükleme
```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="models/qwen-turkish-chat",
    repo_id="your-username/qwen-turkish-chat",
    token="YOUR_HF_TOKEN"
)
```

## 7. Test Etme

```bash
python src/evaluation/test_model.py \
  --model_path models/qwen-turkish-chat \
  --test_file data/splits/test.jsonl
```

## 8. Instance Sonlandırma

⚠️ **ÖNEMLİ:** Training bittikten sonra instance'ı MUTLAKA sonlandırın!

1. Model'i indirin/yükleyin
2. Vast.ai dashboard'dan instance'ı `Destroy`
3. Ücretlendirme durduğunu kontrol edin

## Troubleshooting

### OOM (Out of Memory) Hatası
```yaml
# training_config.yaml'da:
per_device_train_batch_size: 2  # 4'ten 2'ye düşür
gradient_accumulation_steps: 8  # 4'ten 8'e çıkar
```

### Yavaş Training
```bash
# Flash Attention ekle
pip install flash-attn --no-build-isolation

# Mixed precision kontrol et
# training_config.yaml'da bf16: true olmalı
```

### CUDA Versiyonu Uyumsuzluğu
```bash
# PyTorch'u CUDA versiyonuna göre yeniden yükle
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Maliyet Optimizasyonu

1. **Spot Instances:** %50-70 daha ucuz ama kesintiye uğrayabilir
2. **Checkpoint:** Her 100 step'te kayıt (`save_steps: 100`)
3. **Training süresi:** LoRA rank'i düşürerek hızlandırabilirsiniz
4. **Veri miktarı:** İlk denemede 5K örnekle test edin

## Örnek Training Süresi (H200 NVL)

- 10K örnek, 3 epoch, batch 16, seq 2048
- Tahmini süre: **1-1.5 saat**
- Maliyet: **$3-6**

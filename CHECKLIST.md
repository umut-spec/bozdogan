# ✅ Başlangıç Checklist

Projeye başlamadan önce bu adımları tamamlayın.

## 📋 Ön Hazırlık

- [ ] Python 3.9+ kurulu
- [ ] pip güncel (`python -m pip install --upgrade pip`)
- [ ] Git kurulu (projeyi GitHub'a yüklemek için)

## 🔑 API & Hesaplar

- [ ] CiciAPI hesabı oluşturuldu
- [ ] CiciAPI key alındı
- [ ] Vast.ai hesabı oluşturuldu (training için)
- [ ] Vast.ai'ye kredi yüklendi ($10-15 yeterli)

## 🚀 Kurulum

- [ ] Proje klasörüne gidildi: `cd turkish-llm-finetuning`
- [ ] Virtual environment oluşturuldu: `python -m venv venv`
- [ ] Virtual environment aktif edildi:
  - Windows: `venv\Scripts\activate`
  - Linux/Mac: `source venv/bin/activate`
- [ ] Dependencies yüklendi: `pip install -r requirements.txt`
- [ ] `.env` dosyası oluşturuldu: `cp .env.example .env`
- [ ] `.env` dosyasına `CICI_API_KEY` eklendi

## 🧪 Test

- [ ] API bağlantısı test edildi: `python scripts/test_api.py`
- [ ] Bağlantı başarılı (✅ çıktısı alındı)

## 💰 Bütçe Planlaması

- [ ] **Veri üretimi:** 10 yuan (~$1.4) - hazır
- [ ] **GPU kiralama:** $5-6 - Vast.ai hesabında var

## 📝 İsteğe Bağlı (Önerilen)

- [ ] GitHub repository oluşturuldu
- [ ] İlk commit yapıldı
- [ ] HuggingFace hesabı var (model yüklemek için)

## 🎯 Hazır mısınız?

Tüm checkbox'lar işaretli ise, başlayın:

```bash
# 1. Veri üretimi (15-20 dk)
python src/data_generation/generate_synthetic_data.py

# 2. Veri işleme (<5 dk)
python src/data_processing/process_data.py

# 3. Vast.ai'de training (1-2 saat)
# docs/VAST_AI_SETUP.md'yi takip edin
```

## ⚠️ Önemli Notlar

1. **İlk kez yapıyorsanız küçük başlayın:**
   - 50 istek × 10 örnek = 500 örnek
   - 1 epoch training
   - Toplam: <30 dakika, <$1

2. **Checkpoint'leri takip edin:**
   - Her 100 istekte otomatik kayıt
   - Kesinti olursa devam edebilirsiniz

3. **Vast.ai instance'ını unutmayın:**
   - Training bitince MUTLAKA kapatın
   - Aksi halde ücretlendirme devam eder!

4. **Model'i yedekleyin:**
   - Training bitince local'e indirin
   - HuggingFace Hub'a yükleyin

## 📞 Yardım

Sorun mu yaşıyorsunuz?

- API hatası → `scripts/test_api.py` ile test edin
- OOM hatası → `training_config.yaml`'da batch size'ı düşürün
- CUDA hatası → PyTorch versiyonunu kontrol edin

---

**Hazır olduğunuzda:** `QUICKSTART.md` dosyasını takip edin 🚀

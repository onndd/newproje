
# JetX Prediction System

Bu proje, JetX oyunu için yapay zeka tabanlı bir tahmin ve simülasyon sistemi sunar.
İki ana model kullanır:
1. **Model A (Feature Tabanlı):** Geçmiş verilerden istatistiksel özellikler çıkararak CatBoost ile tahmin yapar.
2. **Model B (Desen Hafızası):** Geçmişteki benzer oyun desenlerini (k-NN) bularak tahmin yapar.

## Kurulum

1. Gerekli kütüphaneleri yükleyin:
   ```bash
   pip install -r requirements.txt
   ```

2. `jetx.db` dosyanızı bu dizine kopyalayın.

## Kullanım

### 1. Model Eğitimi (Google Colab veya Lokal)

Modelleri eğitmek için `JetX_Orchestrator.ipynb` dosyasını kullanın.
Bu notebook sırasıyla şunları yapar:
- Veriyi yükler ve eğitim/test olarak ayırır.
- Model A'yı eğitir ve kaydeder (`modelA_p15`, `modelA_p3`, `modelA_x`).
- Model B hafızasını oluşturur ve kaydeder (`modelB_memory`).
- Test seti üzerinde sanal kasa simülasyonu çalıştırır.

### 2. Lokal Tahmin Uygulaması

Modeller eğitildikten sonra, tahmin arayüzünü başlatmak için:

```bash
streamlit run app.py
```

Arayüzde:
- Son gelen X değerini girin.
- "Add Result & Predict" butonuna basın.
- Model A ve Model B'nin tahminlerini ve güven skorlarını görün.

## Dosya Yapısı

- `jetx_project/`: Ana proje paketi.
  - `config.py`: Ayarlar ve kategori aralıkları.
  - `data_loader.py`: Veri yükleme işlemleri.
  - `features.py`: Özellik çıkarımı (Feature Engineering).
  - `categorization.py`: Kategori dönüşümleri.
  - `model_a.py`: CatBoost model işlemleri.
  - `model_b.py`: k-NN model işlemleri.
  - `simulation.py`: Sanal kasa simülasyonu.
- `JetX_Orchestrator.ipynb`: Eğitim ve simülasyon notebook'u.
- `app.py`: Streamlit tahmin arayüzü.
- `requirements.txt`: Gerekli kütüphaneler.

## Notlar

- **1.50 Eşiği:** Yapılandırma, 1.49'u kaybeden, 1.50'yi kazanan taraf olarak kabul edecek şekilde ayarlanmıştır.
- **Veri Yetersizliği:** Model B'nin çalışması için en az 200, Model A'nın tam özellik çıkarımı için en az 500 geçmiş veriye ihtiyaç vardır.

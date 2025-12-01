# JetX Prediction System (AI-Powered)

Bu proje, JetX oyunu için geliştirilmiş, **manipülasyona dayanıklı** ve **yüksek hassasiyetli** bir yapay zeka tahmin sistemidir. Basit istatistiklerin ötesine geçerek, oyunun psikolojik durumunu (HMM), anlık trendleri (LSTM) ve geçmiş desenleri (k-NN) analiz eden bir "Uzmanlar Konseyi" (Ensemble) mimarisi kullanır.

## 🚀 Özellikler ve Mimari

Sistem, tek bir modele güvenmek yerine, farklı güçlü yönleri olan modellerin ortak kararını kullanır:

### 1. Uzman Modeller (The Council)
*   **Model A (CatBoost - GPU):** Geniş özellik seti (200+ feature) ile eğitilmiş, GPU hızlandırmalı ana karar verici.
*   **Model B (k-NN - Hafıza):** Geçmiş 15.000 oyun içindeki en benzer desenleri bulur. (Logaritmik ölçekleme ile 5000x gibi uç değerleri de tanır).
*   **Model C (LSTM - Trend):** Zaman serisi analizi ile anlık trendin yönünü (Yükseliş/Düşüş) tahmin eder.
*   **Model D (LightGBM):** CatBoost'un alternatif görüşü olarak görev yapar (Pasif Uzman).
*   **Model E (MLP - Sinir Ağı):** Farklı bir bakış açısı sunan derin öğrenme katmanı.

### 2. Anti-Manipülasyon Katmanı (The Shield)
*   **HMM (Gizli Markov Modeli):** Oyunun o anki "Rejimini" (Soğuk/Normal/Sıcak) tespit eder. Sadece eğitim verisiyle eğitilerek veri sızıntısı önlenmiştir.
*   **RTP Takibi:** Kasanın (Casino) ne kadar kârda veya zararda olduğunu izleyerek "Hasat Dönemi"ni (Harvest Mode) tahmin etmeye çalışır.
*   **Şok Dalgası Analizi:** 10x+ gibi büyük çarpanlardan sonra gelen "Artçı Şokları" analiz eder.

### 3. Optimizasyon ve Performans
*   **Optuna (Hiperparametre Optimizasyonu):** T4 GPU'nun gücünü kullanarak binlerce farklı parametre kombinasyonunu dener ve en iyisini seçer.
*   **Gerçekçi Simülasyon:**
    *   Kazanma kuralı `True Value > Target` olarak ayarlanmıştır (Eşitlikte kayıp varsayılır).
    *   **TP/FP/TN/FN Analizi:** Sadece genel doğruluğa değil, "Yanlış Pozitif" (Para Kaybettiren Hata) oranına odaklanır.

## 🛠 Kurulum

Proje Google Colab üzerinde çalışacak şekilde optimize edilmiştir.

1.  **Google Colab'ı Açın** ve `JetX_Orchestrator.ipynb` dosyasını yükleyin.
2.  **Runtime Type** ayarını **T4 GPU** olarak seçin.
3.  Notebook'u çalıştırın. Sistem otomatik olarak:
    *   Gerekli kütüphaneleri (`catboost`, `optuna`, `hmmlearn` vb.) kuracaktır.
    *   GitHub'dan en güncel kodları çekecektir.
    *   `jetx.db` veritabanını işleyecektir.

## 📊 Kullanım ve İş Akışı

`JetX_Orchestrator.ipynb` sırasıyla şu adımları gerçekleştirir:

1.  **Veri Hazırlığı:** Veriyi yükler, temizler ve özellik çıkarımı yapar.
2.  **HMM Eğitimi:** Rejim tespiti için HMM modelini eğitir (Data Leakage korumalı).
3.  **Optimizasyon (Optuna):** GPU kullanarak CatBoost için en iyi parametreleri bulur.
4.  **Model Eğitimi:** Tüm uzman modelleri (A, B, C, D, E) eğitir.
5.  **Büyük Final (Simülasyon):**
    *   Son 2000 oyun üzerinde modelleri test eder.
    *   Detaylı Kâr/Zarar ve Güven Dağılımı raporları sunar.
    *   Eğitilen modelleri `models.zip` olarak indirir.

## 📂 Dosya Yapısı

*   `jetx_project/`:
    *   `optimization.py`: Optuna ile GPU tabanlı optimizasyon modülü.
    *   `model_lstm.py`: Veri sızıntısı önlenmiş LSTM mimarisi.
    *   `model_hmm.py`: Rejim tespiti.
    *   `features.py`: Gelişmiş özellik mühendisliği (RTP, Streak, Volatility).
    *   `simulation.py`: Gerçekçi kasa yönetimi ve simülasyon.
    *   `evaluation.py`: Detaylı performans metrikleri.
*   `JetX_Orchestrator.ipynb`: Ana yönetim paneli.

## ⚠️ Önemli Notlar

*   **Yatırım Tavsiyesi Değildir:** Bu proje tamamen eğitim ve araştırma amaçlıdır.
*   **Başarı Oranı:** Genel doğruluktan ziyade **Precision (Kazanma Oranı)** hedeflenmiştir. Hedef, her eli bilmek değil, girilen ellerde %70+ başarı sağlamaktır.

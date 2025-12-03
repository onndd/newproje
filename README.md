# JetX Prediction System (AI-Powered)

Bu proje, JetX oyunu için geliştirilmiş, **manipülasyona dayanıklı** ve **yüksek hassasiyetli** bir yapay zeka tahmin sistemidir. Basit istatistiklerin ötesine geçerek, oyunun psikolojik durumunu (HMM), anlık trendleri (LSTM) ve geçmiş desenleri (k-NN) analiz eden ve bunları bir **Meta-Learner** ile birleştiren gelişmiş bir mimari kullanır.

## 🚀 Özellikler ve Mimari

Sistem, tek bir modele güvenmek yerine, farklı güçlü yönleri olan modellerin ortak kararını kullanır:

### 1. Uzman Modeller (The Council)
*   **Model A (CatBoost - GPU):** Geniş özellik seti (200+ feature) ile eğitilmiş, GPU hızlandırmalı ana karar verici. Hem sınıflandırma (P1.5, P3.0) hem de regresyon (Tahmini X) yapar.
*   **Model B (k-NN - Hafıza):** "Tarih tekerrürden ibarettir" prensibiyle çalışır. Geçmiş 15.000 oyun içindeki en benzer desenleri bulur (PCA destekli).
*   **Model C (LSTM - Trend):** Zaman serisi analizi ile son 200 oyunluk periyotları inceleyerek anlık trendin yönünü tahmin eder.
*   **Model D (LightGBM):** CatBoost'un alternatif görüşü olarak görev yapar (Pasif Uzman).
*   **Model E (MLP - Sinir Ağı):** Sadece ham verilerle (Raw Lags) beslenen, insan müdahalesi olmayan "saf" bir bakış açısı sunar.

### 2. Orkestrasyon (The Meta-Learner)
Tüm uzmanların görüşleri, bir **Logistic Regression Meta-Learner** tarafından ağırlıklandırılır. Bu katman, hangi modelin hangi piyasa koşulunda (Soğuk/Sıcak) daha başarılı olduğunu öğrenir ve nihai kararı verir.

### 3. Anti-Manipülasyon Katmanı (The Shield)
*   **Causal HMM (Gizli Markov Modeli):** Oyunun o anki "Rejimini" (Soğuk/Normal/Sıcak) tespit eder. **Causal Prediction** (Nedensel Tahmin) yöntemiyle, geleceği görmeden (lookahead bias olmadan) sadece geçmiş veriye dayanarak anlık durum tespiti yapar.
*   **RTP Takibi:** Kasanın (Casino) ne kadar kârda veya zararda olduğunu izleyerek "Hasat Dönemi"ni (Harvest Mode) tahmin etmeye çalışır.
*   **Şok Dalgası Analizi:** 10x+ gibi büyük çarpanlardan sonra gelen "Artçı Şokları" analiz eder.

### 4. Optimizasyon ve Simülasyon
*   **Optuna (Hiperparametre Optimizasyonu):** T4 GPU'nun gücünü kullanarak binlerce farklı parametre kombinasyonunu dener.
*   **Gelişmiş Simülasyon (4 Farklı Strateji):**
    *   **Kasa 1 (Conservative):** 1.50x hedef, %75+ güven.
    *   **Kasa 2 (Moderate):** 1.50x hedef, %85+ güven (Daha seçici).
    *   **Kasa 3 (High Risk):** 3.00x ve üzeri hedefler için fırsat kollar.
    *   **Kasa 4 (Smart Kelly):** Kelly Kriteri'ne dayalı dinamik bahis yönetimi. Güven arttıkça bahsi artırır, riskli durumlarda bahsi kısar.

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
4.  **Model Eğitimi:** Tüm uzman modelleri (A, B, C, D, E) ve Meta-Learner'ı eğitir.
5.  **Büyük Final (Simülasyon):**
    *   Son test verisi üzerinde 4 farklı kasa stratejisini yarıştırır.
    *   Detaylı Kâr/Zarar, Drawdown ve Güven Dağılımı raporları sunar.
    *   Eğitilen modelleri `models.zip` olarak indirir.

## 📂 Dosya Yapısı

*   `jetx_project/`:
    *   `features.py`: Gelişmiş özellik mühendisliği (RTP, Streak, Volatility).
    *   `ensemble.py`: Meta-Learner ve model birleştirme mantığı.
    *   `simulation.py`: 4 farklı strateji ile gerçekçi kasa yönetimi.
    *   `optimization.py`: Optuna ile GPU tabanlı optimizasyon.
    *   `model_a.py`: CatBoost (Ana Model).
    *   `model_b.py`: k-NN (Hafıza Modeli).
    *   `model_c.py`: LSTM (Trend Modeli).
    *   `model_d.py`: LightGBM.
    *   `model_e.py`: MLP (Sinir Ağı).
    *   `model_hmm.py`: Rejim tespiti.
    *   `evaluation.py`: Detaylı performans metrikleri.
*   `JetX_Orchestrator.ipynb`: Ana yönetim paneli.

## ⚠️ Önemli Notlar

*   **Yatırım Tavsiyesi Değildir:** Bu proje tamamen eğitim ve araştırma amaçlıdır.
*   **Başarı Oranı:** Genel doğruluktan ziyade **Precision (Kazanma Oranı)** hedeflenmiştir. Hedef, her eli bilmek değil, girilen ellerde %70+ başarı sağlamaktır.

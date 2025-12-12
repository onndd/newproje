# 🚀 JetX Yüksek Başarımlı Tahmin Sistemi (AI-Powered)

Bu proje, JetX oyun sonuçlarını tahmin etmek için geliştirilmiş, **Ensemble Learning (Topluluk Öğrenmesi)** ve **Derin Öğrenme** teknolojilerini birleştiren son teknoloji bir yapay zeka sistemidir.

Proje, **%70 - %80** aralığında doğruluk (Target > 1.50x) hedefler ve gelişmiş sinyal işleme teknikleri ile piyasa rejimini (HMM) analiz eder.

---

## 🌟 Öne Çıkan Özellikler

### 🧠 1. Çoklu Model Mimarisi (Ensemble)
Tek bir model yerine, 6 farklı uzmanın ortak kararını kullanırız:
*   **Model A (CatBoost):** Tablosal veri ve özellik mühendisliği uzmanı.
*   **Model B (Memory/k-NN):** Geçmiş oyun desenlerini (pattern) hatırlayan hafıza modülü.
*   **Model C (LSTM):** Zaman serisindeki sıralı ilişkileri çözen Derin Öğrenme ağı.
*   **Model D (LightGBM):** Hızlı ve hafif karar ağacı tabanlı model.
*   **Model E (MLP):** Karmaşık lineer olmayan ilişkileri öğrenen Yapay Sinir Ağı.
*   **Model F (Transformer):** "Attention" mekanizması ile uzun vadeli bağımlılıkları analiz eden modern mimari.

### 🤖 2. Meta-Learner (Orkestra Şefi)
Tüm alt modellerin tahminlerini ve piyasa durumunu (HMM) toplayarak son kararı veren, hataya dayanıklı (robust) bir üst modeldir. Eksik veri veya model olsa bile çökmeyerek "Nötr" modda çalışmaya devam eder.

### 📊 3. HMM Piyasa Analizi
**Hidden Markov Model (HMM)** ile piyasanın o anki "Ruh Hali" tespit edilir:
*   ❄️ **Cold (Düşük):** Piyasa durgun, riskli.
*   🌤️ **Normal:** Standart akış.
*   🔥 **Hot (Yüksek):** Yüksek çarpanların sık geldiği kazançlı dönem.

### 📥 4. Toplu Veri Girişi (Yeni!)
Oyunu her an takip edemeseniz bile, geçmiş verileri topluca sisteme yükleyebilirsiniz. Sistem, verileri otomatik olarak temizler ve kronolojik sıraya dizerek veritabanına işler.

---

## 🛠️ Kurulum

Proje **Python 3.8+** gerektirir. Önerilen kurulum adımları:

1.  **Depoyu Klonlayın:**
    ```bash
    git clone https://github.com/onndd/newproje.git
    cd newproje
    ```

2.  **Sanal Ortam Oluşturun (Önerilen):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Mac/Linux
    # .venv\Scripts\activate   # Windows
    ```

3.  **Bağımlılıkları Yükleyin:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 🚀 Kullanım

### 1. Modellerin Eğitimi (Orchestrator)
Sistemi sıfırdan kuruyorsanız veya veritabanı büyüdüyse modelleri yeniden eğitmelisiniz.
*   **Dosya:** `JetX_Orchestrator.ipynb`
*   **Kullanım:** Jupyter Notebook veya Google Colab ile açın. `Run All` yaparak veri temizliği, eğitim ve simülasyon adımlarını otomatik tamamlayın.
*   **Optuna:** Hiperparametre optimizasyonu için notebook içindeki "4.1 MODEL OPTİMİZASYONU" hücresini kullanabilirsiniz.

### 2. Tahmin Uygulaması (Streamlit)
Canlı tahmin arayüzünü başlatmak için:
```bash
streamlit run app.py
```

#### Arayüz Modları:
*   **🚀 Canlı Tahmin:**
    *   Son gelen çarpanı (X) kutuya girin.
    *   `Add Result & Predict` butonuna basın.
    *   Sistem veriyi kaydeder, analiz eder ve **BET (Oyna)** veya **WAIT (Bekle)** sinyali verir.
    
*   **📥 Toplu Veri Girişi:** (Sol Menüden Seçin)
    *   Excel veya geçmiş listesinden kopyaladığınız verileri (örneğin son 50 oyun) kutuya yapıştırın.
    *   **En Üst = En Yeni**, **En Alt = En Eski** olacak şekilde yapıştırın.
    *   Sistem otomatik olarak listeyi temizler ve doğru sırayla veritabanına ekler.

---

## 📂 Dosya Yapısı

*   `app.py`: Ana uygulama (Streamlit). Arayüz ve tahmin mantığı.
*   `JetX_Orchestrator.ipynb`: Eğitim, test ve simülasyon merkezi.
*   `jetx_project/`:
    *   `features.py`: Özellik çıkarımı (Feature Engineering). **(Sızıntı Korumalı)**
    *   `ensemble.py`: Meta-Learner ve oylama mantığı.
    *   `model_*.py`: Model tanımları (LSTM, Transformer, CatBoost vb.).
    *   `config.py`: Ayarlar ve sabitler.
*   `jetx.db`: Oyun verilerinin tutulduğu SQLite veritabanı.

---

## ⚠️ Kritik Notlar & Feragatname

1.  **Sızıntı Koruması (No Data Leakage):** Proje, eğitim sırasında geleceği görmeyi (look-ahead bias) engelleyen katı kurallarla yazılmıştır. Bu nedenle eğitim skorları "yapay" olarak yüksek çıkmaz, gerçeği yansıtır.
2.  **Yatırım Tavsiyesi Değildir:** Bu proje tamamen eğitim ve araştırma amaçlıdır. Kumar veya bahis oynamayı teşvik etmez. Oluşabilecek maddi kayıplardan geliştirici sorumlu tutulamaz.
3.  **1.50x Kuralı:** Sistem 1.50x çarpanını (veya 1.57x güvenli çıkış) hedefler. Daha yüksek riskli çarpanlar için tasarlanmamıştır.

---

**Geliştirici:** Numan Öndeş  
**Lisans:** MIT

---

## 🧪 Testler ve Doğrulama
Bu proje, kod kalitesini ve sistem sağlığını doğrulamak için otomatik test altyapısına sahiptir.

### 1. Birim Testleri (Pytest)
Veritabanı bağlantısı ve model tahmin fonksiyonlarını test etmek için:
```bash
pytest
```

### 2. Smoke Test (Hızlı Kontrol)
Sistemin uçtan uca (DB -> Model -> Tahmin) çalışıp çalışmadığını tek komutla görmek için:
```bash
python run_smoke_test.py
```
*Bu komut, Streamlit arayüzünü açmadan arka planda tüm sistemi kontrol eder.*

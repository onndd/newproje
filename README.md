# JetX Tahmin Sistemi (Streamlit + Ensemble)

Bu proje, JetX oyun sonuçlarını tahmin etmek için birden fazla makine öğrenmesi modelini (Ensemble) birleştiren kapsamlı bir Streamlit uygulamasıdır.

## 🎯 Temel Hedef: 1.50x Eşiği
Sistemin birincil amacı, bir sonraki çarpanın **1.50x'in ÜZERİNDE mi yoksa ALTINDA mı** olacağını tahmin etmektir.
- **Neden 1.50x?** Bu bizim kritik karlılık sınırımızdır.
- **1.50x Üstü:** Hedef Bölge (Kazan).
- **1.50x Altı:** Kayıp Bölgesi (Uzak Dur).
- **Strateji:** Sistem muhafazakar olacak şekilde tasarlanmıştır. Sadece sonucun 1.50x'i geçeceğinden **yüksek derecede eminse (>%75)** "BAHİS YAP" sinyali üretir.

## 📊 Eklenen Metrikler (ROC-AUC ve Kar/Zarar)
Model performansını ölçmek için eklenen **ROC-AUC** ve **Kar/Zarar (Profit/Loss)** metrikleri, **1.50x ve 3.00x eşikleri** için hesaplanmaktadır.
- **Ham X Değeri Değil:** Bu metrikler, modelin "Tam olarak kaç x gelecek?" (Regresyon) tahminini değil, "1.50x'i geçer mi?" (Sınıflandırma) başarısını ölçer.
- **Kar/Zarar Simülasyonu:** Modelin her "Oyna" dediğinde 1 birim bahis yaptığımızı varsayarak, gerçekte ne kadar kazanıp kaybedeceğimizi simüle eder.

## ⏳ Kronolojik Bütünlük (Veri Sızıntısı Yok)
Gerçekçi performans sonuçları elde etmek için bu proje **Zaman Serisi Doğrulama (Time-Series Validation)** ilkelerine sıkı sıkıya bağlıdır:
- **Karıştırma Yok (No Shuffling):** Veriler ASLA karıştırılmaz. Olayların sırası, gerçekleştiği gibi aynen korunur.
- **Sıkı Bölme (Strict Splitting):**
    - **Eğitim (Train):** Geçmiş verilerin ilk %70'i.
    - **Boşluk (Gap):** %5'lik bir tampon bölge, sızıntıyı önlemek için kullanılmadan bırakılır.
    - **Doğrulama (Validation):** Sonraki %15'lik kısım.
    - **Test:** Son %10 (en güncel veriler).
- **Neden?** Gerçek zamanlı bahiste geleceği göremeyiz. Verileri karıştırmak, modelin gelecekteki desenleri görerek "kopya çekmesine" neden olur. Bizim katı yaklaşımımız, test sonuçlarının canlı ortamdaki gerçek performansı yansıtmasını garanti eder.

## 🏗️ Mimari Bileşenler
- **Model A (CatBoost):** Zengin özellik seti ile 1.5x / 3.0x olasılığı ve beklenen X regresyonu.
- **Model B (k-NN / Hafıza):** 300 oyunluk geçmiş desen benzerliği ve PCA ile hızlı sorgu.
- **Model C (LSTM):** 200 adımlık dizilerden zaman serisi trendlerini yakalama.
- **Model D (LightGBM):** Hızlı ve hafif gradyan artırma modeli.
- **Model E (MLP):** Ham verilerle çalışan Yapay Sinir Ağı.
- **Model F (Transformer):** "Attention" mekanizması ile uzun vadeli ilişkileri çözen modern mimari.
- **HMM (Gizli Markov Modeli):** Piyasanın "Ruh Halini" (Volatilite Durumunu) analiz eder.
- **Meta-Learner:** Tüm bu modellerin tahminlerini alıp son kararı veren "Beyin".

## Çalışma Akışı (app.py)
1) Uygulama açıldığında `jetx.db` varsa son 2000 kayıt RAM’e alınır (OOM koruması).  
2) Kullanıcı yeni sonucu girer, önce SQLite’a yazılır, sonra RAM geçmişi güncellenir.  
3) Özellikler: 500+ geçmiş varsa Model A/D/E için feature engineering; 300+ için k-NN, 200+ için LSTM/Transformer dizileri hazırlanır.  
4) HMM son 500 oyundan rejim çıkarır.  
5) Meta-learner, alt model olasılıkları + HMM + 1.00x frekansını alır ve **1.50x için nihai olasılığı** döner. 0.65 üstünde “BET” sinyali, aksi halde “WAIT”.  
6) Tüm modeller yüklenemezse uygulama durur; eksik modeller için ekranda hata görülür.

## Kurulum ve Çalıştırma
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

## Modellerin Eğitimi
- Notebook: `JetX_Orchestrator.ipynb` (GPU önerilir).  
- Çıkışlar: `modelA_*`, `modelB_memory`, `modelC_*`, `modelD_*`, `modelE_*`, `model_transformer.h5`, `model_hmm.pkl`, `meta_learner.pkl` aynı dizinde saklanır.  
- Meta-learner Transformer’lı eğitildiyse inference sırasında Transformer modelinin de yüklenmesi gerekir (aksi halde varsayılan 0.5 ile doldurulur).

## Dosya Yapısı (özet)
- `app.py`: Streamlit arayüzü, tahmin akışı, SQLite yazma/okuma.
- `jetx_project/features.py`: Feature engineering.
- `jetx_project/model_*`: Her alt modelin eğitim/yükleme mantığı.
- `jetx_project/ensemble.py`: Meta feature hazırlanması ve meta-learner tahmini.
- `jetx_project/data_loader.py`: Veritabanından veriyi parça parça okuma (limit desteği).
- `verify_fixes.py`: Basit veri yükleme testi (dummy DB ile).

## Kritik Notlar
- **1.50x eşiği korunmalıdır:** Eşik sabit; meta-learner ve sinyalleme bu hedef için tasarlandı.
- Kayıt sayısı azsa (<500) tahmin yapılmaz; kullanıcıya uyarı verilir.
- Varsayılan fallback ortalaması sadece meta-learner yoksa devrededir; gerçek kullanım için modellerin eğitilmiş olması gerekir.

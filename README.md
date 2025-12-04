# JetX Prediction System (Streamlit + Ensemble)

JetX oyun sonuçlarını tahmin etmek için birden fazla modeli birleştiren bir Streamlit uygulaması. 1.50x eşiği ana hedef; tüm modeller ve meta-learner bu eşiğe göre optimize edilmiştir.

## Mimaride Neler Var?
- **Model A (CatBoost):** Zengin feature seti ile 1.5 / 3.0 olasılığı ve beklenen X regresyonu.
- **Model B (k-NN / Hafıza):** 300 oyunluk desen benzerliği ve PCA ile hızlı sorgu.
- **Model C (LSTM):** 200 adımlık dizilerden trend yakalama.
- **Model D (LightGBM):** Hafif, ağaç tabanlı alternatif.
- **Model E (MLP):** Sadece ham lag + HMM ile çeşitlilik katar.
- **Model T (Transformer):** Uzun bağımlılıkları dikkat (attention) katmanıyla öğrenir.
- **HMM (Categorical/GMM):** Piyasa rejimi (Cold/Normal/Hot) tespiti.
- **Meta-Learner (LogReg):** A, B, C, D, E, T ve HMM çıktılarından nihai 1.5x olasılığını üretir.

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

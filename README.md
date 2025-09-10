
# 🎬 IMDB Comment Tagging with LSTM

Bu proje, **IMDB film yorumlarının duygu analizi (sentiment analysis)** için geliştirilmiş bir **Makine Öğrenmesi & Derin Öğrenme** çalışmasıdır.  
Amaç, kullanıcı yorumlarını **pozitif** ve **negatif** olarak sınıflandırmak için LSTM (Long Short-Term Memory) tabanlı bir model kurmaktır.

## 👥 Katkı Sağlayanlar

- **Dilara Çelik**  
  - Veri hazırlama ve temizleme  
  - Model mimarisi geliştirme (LSTM)  
  - Eğitim/validasyon süreçleri  
  - Sonuç analizi ve raporlama
  - Yazılım geliştirme süreci

- **Furkan Yiğit**  
  - Araştırma, test ve deneme süreçleri  
  - Makine öğrenmesi ve derin öğrenme yöntemlerinin belirlenmesi  
  - Parametre ayarlamaları, performans karşılaştırmaları  
  - Model sonuçlarının değerlendirilmesi
  - Yazılım geliştirme süreci



## 🧠 Kullanılan Yöntemler
- **Doğal Dil İşleme (NLP):** IMDB yorumlarının ön işlenmesi  
  - Tokenizasyon, padding, stopword temizleme  
- **Derin Öğrenme (LSTM):** Yorumların dizisel yapısını yakalamak için LSTM tabanlı RNN modeli  
- **Makine Öğrenmesi Testleri:** Karşılaştırmalı olarak ML yöntemlerinin (Naive Bayes, Logistic Regression vb.) uygulanması  
- **Performans Ölçütleri:** Accuracy, Precision, Recall, F1-Score  

## Özellikler
- `Eğitimi Başlat` butonu ile yeniden eğitim
- Doğruluk, F1, Kesinlik, Hassaslık, AUC metrikleri
- Kayıp/Doğruluk, ROC, Konfüzyon Matrisi grafikleri
- Örnek incelemelerin (index 22 ve 13) çözülmüş hali ve tahmini
- Metrikler ve grafikler her eğitim sonunda güncellenir

> Not: IMDB veri seti ilk çalıştırmada otomatik indirilir ve yerelde önbelleğe alınır.

## 📂 Proje Yapısı
```

IMDB_CommentTag_LSTM/
├─ data/ # IMDB dataset (ham/veri işlenmiş halleri)
├─ notebooks/ # Jupyter Notebook deneyleri
├─ models/ # Eğitilmiş model dosyaları
├─ results/ # Sonuç raporları, görselleştirmeler
├─ main.py # Çalıştırılabilir script
└─ README.md # Bu dosya
```

## Kurulum

```bash
cd imdb_lstm_webapp
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
python app.py
```

Tarayıcıda `http://127.0.0.1:5000` adresine gidin.

📊 Sonuçlar

- LSTM tabanlı model, klasik ML yöntemlerine göre daha yüksek doğruluk sağlamıştır.
- Eğitim sürecinde overfitting azaltmak için dropout ve düzenlileştirme yöntemleri kullanılmıştır.
- Nihai model ile IMDB yorumlarının sınıflandırılmasında başarılı bir performans elde edilmiştir.

🛠️ Yol Haritası

 - Attention mekanizmaları ile LSTM performansını geliştirme
 - Bidirectional LSTM (BiLSTM) denemeleri
 - Transformer tabanlı modeller (BERT, RoBERTa) ile karşılaştırma
 - Daha geniş veri setleri ile test etme


## 🤝 Katkı

Katkılar memnuniyetle kabul edilir!
- Fork’la
- Branch aç: feature/isim
- Değişikliklerini yap + test et
- PR aç: değişiklik özetini ve nedenini yaz

## 📜 Lisans
- MIT © 2025 Dilara Çelik
- MIT © 2025 Furkan Yiğit






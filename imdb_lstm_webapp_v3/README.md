
# IMDB LSTM – Tek Sayfa Eğitim & Görselleştirme

Bu proje, TensorFlow ile IMDB yorum sınıflandırma modelini (LSTM) eğitir ve sonuçları **tek sayfalık** bir HTML arayüzünde gösterir.

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

## Özellikler
- `Eğitimi Başlat` butonu ile yeniden eğitim
- Doğruluk, F1, Kesinlik, Hassaslık, AUC metrikleri
- Kayıp/Doğruluk, ROC, Konfüzyon Matrisi grafikleri
- Örnek incelemelerin (index 22 ve 13) çözülmüş hali ve tahmini
- Metrikler ve grafikler her eğitim sonunda güncellenir

> Not: IMDB veri seti ilk çalıştırmada otomatik indirilir ve yerelde önbelleğe alınır.

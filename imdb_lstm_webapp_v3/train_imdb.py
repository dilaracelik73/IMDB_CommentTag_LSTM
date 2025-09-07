
import os, json
from pathlib import Path
from datetime import datetime

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _decode_review(text, reverse_word_index):
    # text: list/array of token ids (possibly padded); imdb uses offsets by 3
    return ' '.join([reverse_word_index.get(int(i) - 3, '?') for i in list(text)])

def _build_model(vocab_size=10000):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=64),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def _plot_training(history, plots_dir: Path):
    plots_dir.mkdir(parents=True, exist_ok=True)
    # Loss
    fig1 = plt.figure()
    plt.plot(history.history['loss'], label='Eğitim Kaybı')
    plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
    plt.xlabel('Epoch'); plt.ylabel('Kayıp'); plt.title('Model Kayıpları'); plt.legend()
    fig1.tight_layout(); fig1.savefig(plots_dir / "loss.png", dpi=150)
    plt.close(fig1)

    # Accuracy
    fig2 = plt.figure()
    plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
    plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
    plt.xlabel('Epoch'); plt.ylabel('Doğruluk'); plt.title('Model Doğruluğu'); plt.legend()
    fig2.tight_layout(); fig2.savefig(plots_dir / "accuracy.png", dpi=150)
    plt.close(fig2)

def _plot_roc_and_cm(y_true, y_prob, y_pred, plots_dir: Path):
    plots_dir.mkdir(parents=True, exist_ok=True)
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig3 = plt.figure()
    plt.plot(fpr, tpr, label=f'ROC Eğrisi (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('Yanlış Pozitif Oranı'); plt.ylabel('Doğru Pozitif Oranı'); plt.title('ROC Eğrisi'); plt.legend()
    fig3.tight_layout(); fig3.savefig(plots_dir / "roc.png", dpi=150)
    plt.close(fig3)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    fig4 = plt.figure(figsize=(4.8, 4.8))
    im = plt.imshow(cm, interpolation='nearest')
    plt.title('Konfüzyon Matrisi'); plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Negatif', 'Pozitif'])
    plt.yticks(tick_marks, ['Negatif', 'Pozitif'])
    thresh = cm.max() / 2.0 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('Gerçek Etiketler'); plt.xlabel('Tahmin Edilen Etiketler')
    fig4.tight_layout(); fig4.savefig(plots_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig4)

def train_and_evaluate(epochs=5, batch_size=64, maxlen=500, vocab_size=10000,
                       artifacts_dir="artifacts", plots_dir="static/plots", sample_indices=(22, 13)):
    """
    Train and evaluate the IMDB LSTM model.
    sample_indices: tuple/list of integer indices to decode & predict from the TRAIN split after padding.
    """
    artifacts_dir = Path(artifacts_dir); artifacts_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = Path(plots_dir); plots_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset with limited vocab
    (train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.imdb.load_data(num_words=vocab_size)

    # 80/20 split (re-split)
    train_data, test_data, train_labels, test_labels = train_test_split(
        train_data, train_labels, test_size=0.2, random_state=42
    )

    # Word index for decoding
    word_index = tf.keras.datasets.imdb.get_word_index()
    reverse_word_index = {value: key for key, value in word_index.items()}

    # Pad sequences
    train_data = tf.keras.preprocessing.sequence.pad_sequences(train_data, maxlen=maxlen)
    test_data  = tf.keras.preprocessing.sequence.pad_sequences(test_data,  maxlen=maxlen)

    # Build & train model
    model = _build_model(vocab_size=vocab_size)
    history = model.fit(train_data, np.array(train_labels), epochs=epochs, batch_size=batch_size,
                        validation_split=0.2, verbose=1)

    # Evaluate
    test_loss, test_accuracy = model.evaluate(test_data, np.array(test_labels), verbose=0)

    # Predictions & metrics
    probs = model.predict(test_data, verbose=0).reshape(-1)
    y_pred = (probs > 0.5).astype(int)
    y_true = np.array(test_labels).astype(int)

    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    auc = roc_auc_score(y_true, probs)
    class_counts = np.bincount(y_true, minlength=2).tolist()

    # Plots
    _plot_training(history, plots_dir)
    _plot_roc_and_cm(y_true, probs, y_pred, plots_dir)

    # Decode requested samples
    # Sanitize sample indices
    samples_out = {}
    total = len(train_data)
    idxs = []
    try:
        if isinstance(sample_indices, (list, tuple)):
            idxs = [int(x) for x in sample_indices]
        else:
            idxs = [int(sample_indices)]
    except Exception:
        idxs = [22, 13]

    for idx in idxs:
        if 0 <= idx < total:
            seq = train_data[idx]
            decoded = _decode_review(seq, reverse_word_index)
            pred_prob = float(model.predict(np.expand_dims(seq, 0), verbose=0).reshape(-1)[0])
            samples_out[str(idx)] = {
                "decoded_text": decoded,
                "pred_probability": pred_prob,
                "pred_label": "Pozitif" if pred_prob > 0.5 else "Negatif"
            }
        else:
            samples_out[str(idx)] = {
                "decoded_text": f"(Uyarı) İndeks {idx} geçersiz. 0–{total-1} aralığında bir değer giriniz.",
                "pred_probability": None,
                "pred_label": "—"
            }

    # Save model and metrics
    model_dir = Path("model"); model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "imdb_lstm.h5"
    model.save(model_path)

    metrics = {
        "status": "ok",
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "params": {"epochs": int(epochs), "batch_size": int(batch_size), "maxlen": int(maxlen), "vocab_size": int(vocab_size)},
        "results": {
            "test_loss": float(test_loss),
            "test_accuracy": float(test_accuracy),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "auc": float(auc),
            "class_counts": class_counts
        },
        "samples": samples_out
    }

    with open(Path(artifacts_dir) / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return metrics

def load_metrics(path):
    path = Path(path)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

if __name__ == "__main__":
    train_and_evaluate()

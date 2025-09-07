
import os, json, time
from pathlib import Path
from flask import Flask, render_template, jsonify, request
from train_imdb import train_and_evaluate, load_metrics

BASE_DIR = Path(__file__).resolve().parent
PLOTS_DIR = BASE_DIR / "static" / "plots"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/metrics", methods=["GET"])
def metrics():
    m = load_metrics(ARTIFACTS_DIR / "metrics.json")
    if m is None:
        m = {"status": "empty"}
    ts = int(time.time())
    m["plots"] = {
        "loss": f"/static/plots/loss.png?v={ts}",
        "accuracy": f"/static/plots/accuracy.png?v={ts}",
        "roc": f"/static/plots/roc.png?v={ts}",
        "cm": f"/static/plots/confusion_matrix.png?v={ts}"
    }
    return jsonify(m)

@app.route("/train", methods=["POST"])
def train():
    body = request.get_json(silent=True) or {}
    epochs = int(body.get("epochs", 5))
    batch_size = int(body.get("batch_size", 64))
    maxlen = int(body.get("maxlen", 500))
    vocab_size = int(body.get("vocab_size", 10000))
    s1 = int(body.get("sample1", 22))
    s2 = int(body.get("sample2", 13))

    # Backward compatible call (in case an older train_imdb.py is in use)
    try:
        metrics = train_and_evaluate(epochs=epochs, batch_size=batch_size, maxlen=maxlen,
                                     vocab_size=vocab_size, artifacts_dir=str(ARTIFACTS_DIR),
                                     plots_dir=str(PLOTS_DIR), sample_indices=(s1, s2))
    except TypeError:
        metrics = train_and_evaluate(epochs=epochs, batch_size=batch_size, maxlen=maxlen,
                                     vocab_size=vocab_size, artifacts_dir=str(ARTIFACTS_DIR),
                                     plots_dir=str(PLOTS_DIR))

    ts = int(time.time())
    metrics["plots"] = {
        "loss": f"/static/plots/loss.png?v={ts}",
        "accuracy": f"/static/plots/accuracy.png?v={ts}",
        "roc": f"/static/plots/roc.png?v={ts}",
        "cm": f"/static/plots/confusion_matrix.png?v={ts}"
    }
    return jsonify(metrics)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

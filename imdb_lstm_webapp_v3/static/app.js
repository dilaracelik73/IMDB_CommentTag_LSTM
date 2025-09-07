
let classChart = null;

function fmt(n) {
  if (n === undefined || n === null || isNaN(n)) return "—";
  return Number(n).toFixed(4);
}
function fmt2(n) {
  if (n === undefined || n === null || isNaN(n)) return "—";
  return Number(n).toFixed(2);
}

function fmtDateHuman(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  if (isNaN(d)) return iso; // backend farklı format verirse olduğu gibi göster
  return d.toLocaleString("tr-TR", {
    year: "numeric", month: "2-digit", day: "2-digit",
    hour: "2-digit", minute: "2-digit", second: "2-digit"
  });
}

function renderParamsPretty(p) {
  // p obje olabilir veya JSON string
  let obj = p;
  if (typeof p === "string") {
    try { obj = JSON.parse(p); } catch { /* bırak */ }
  }
  if (typeof obj !== "object" || obj === null) return p || "—";

  const labels = {
    batch_size: "Batch Size",
    epochs: "Epochs",
    maxlen: "Maxlen",
    vocab_size: "Vocab Size"
  };

  const rows = Object.keys(obj).map(k => {
    const label = labels[k] || k;
    const val = obj[k];
    return `<li><span>${label}</span><b>${val}</b></li>`;
  }).join("");

  return `<ul class="param-list">${rows}</ul>`;
}


async function fetchMetrics() {
  const r = await fetch("/metrics");
  const m = await r.json();
  if (!m || m.status === "empty") {
    document.getElementById("status").textContent = "Hazır metrik yok. Eğitimi başlatın.";
    return;
  }
  const res = m.results || {};
  document.getElementById("acc").textContent = fmt(res.test_accuracy);
  document.getElementById("f1").textContent = fmt(res.f1);
  document.getElementById("precision").textContent = fmt(res.precision);
  document.getElementById("recall").textContent = fmt(res.recall);
  document.getElementById("auc").textContent = fmt(res.auc);
  document.getElementById("trainedAt").textContent = fmtDateHuman(m.trained_at);
  document.getElementById("params").innerHTML = renderParamsPretty(m.params);


  // plots
  if (m.plots) {
    document.getElementById("lossImg").src = m.plots.loss;
    document.getElementById("accImg").src = m.plots.accuracy;
    document.getElementById("rocImg").src = m.plots.roc;
    document.getElementById("cmImg").src = m.plots.cm;
  }
// class chart
const counts = (res.class_counts || [0, 0]);
const ctx = document.getElementById("classChart").getContext("2d");

// Renkler
const styles = getComputedStyle(document.documentElement);
const TEXT   = (styles.getPropertyValue("--text")  || "#fff").trim();      
const GRID   = "rgba(255,255,255,0.06)";                                   
const BORDER = "rgba(255,255,255,0.12)";                                   
const NEG = "hsl(0, 100%, 35%)";   
const POS = "hsl(160, 80%, 30%)";  

if (classChart) classChart.destroy();
classChart = new Chart(ctx, {
  type: "bar",
  data: {
    labels: ["Negatif", "Pozitif"],
    datasets: [{
      label: "Yorum Sayısı",
      data: counts,
      backgroundColor: [NEG, POS],
      borderColor: [NEG, POS],
      borderWidth: 1,
      borderRadius: 6
    }]
  },
  options: {
    responsive: true,
    scales: {
      x: {
        ticks: { color: TEXT },
        grid:  { color: GRID },
        border:{ color: BORDER }
      },
      y: {
        beginAtZero: true,
        ticks: { color: TEXT },
        grid:  { color: GRID },
        border:{ color: BORDER }
      }
    },
    plugins: {
      legend: { labels: { color: TEXT } },     
      tooltip: {
        backgroundColor: "rgba(11,15,20,0.95)",
        titleColor: TEXT,
        bodyColor: TEXT,
        borderColor: BORDER,
        borderWidth: 1
      }
    }
  }
});


  // samples
  const samplesDiv = document.getElementById("samples");
  samplesDiv.innerHTML = "";
  const samples = m.samples || {};
  Object.keys(samples).forEach(k => {
    const s = samples[k];
    const badgeClass = (s.pred_label === "Pozitif") ? "pos" : "neg";
    const el = document.createElement("div");
    el.className = "sample";
    el.innerHTML = `
      <div class="sample-header">
        <strong>Index ${k}</strong>
        <span class="badge ${badgeClass}">${s.pred_label} (${fmt2(s.pred_probability)})</span>
      </div>
      <div class="sample-text">${(s.decoded_text || "").replaceAll("<", "&lt;").replaceAll(">", "&gt;")}</div>
    `;
    samplesDiv.appendChild(el);
  });
}

async function train() {
  const status = document.getElementById("status");
  status.textContent = "Eğitim başlatıldı... (bu işlem biraz sürebilir)";
  const body = {
    epochs: parseInt(document.getElementById("epochs").value || "5"),
    batch_size: parseInt(document.getElementById("batch").value || "64"),
    maxlen: parseInt(document.getElementById("maxlen").value || "500"),
    vocab_size: parseInt(document.getElementById("vocab").value || "10000"),
    sample1: parseInt(document.getElementById("sample1").value || "13"),
    sample2: parseInt(document.getElementById("sample2").value || "22")
  };
  try {
    const r = await fetch("/train", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (!r.ok) throw new Error("HTTP " + r.status);
    const m = await r.json();
    status.textContent = "Eğitim tamamlandı ✓";
    // refresh everything
    await fetchMetrics();
  } catch (e) {
    console.error(e);
    status.textContent = "Eğitim sırasında hata oluştu: " + e.message;
  }
}

document.getElementById("trainBtn").addEventListener("click", train);
fetchMetrics();



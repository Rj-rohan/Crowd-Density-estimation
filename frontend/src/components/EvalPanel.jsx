import { useState } from "react";

export default function EvalPanel() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const runEval = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("http://localhost:8000/evaluate?part=A");
      const data = await res.json();
      if (data.error) setError(data.error);
      else setResult(data);
    } catch {
      setError("Backend unreachable");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card eval-card">
      <div className="eval-header">
        <h3>🧪 Model Evaluation</h3>
        <button className="btn btn-primary" onClick={runEval} disabled={loading}>
          {loading ? "Running…" : "Run Evaluation"}
        </button>
      </div>

      {error && <p className="eval-error">{error}</p>}

      {result && (
        <div className="eval-grid">
          <EvalItem label="MAE" value={result.mae} desc="Mean Absolute Error" color="#6366f1" />
          <EvalItem label="MSE" value={result.mse} desc="Mean Squared Error" color="#f59e0b" />
          <EvalItem label="Samples" value={result.samples} desc="Test images evaluated" color="#22c55e" />
        </div>
      )}

      {!result && !error && (
        <p className="muted">Click "Run Evaluation" after model training completes.</p>
      )}
    </div>
  );
}

function EvalItem({ label, value, desc, color }) {
  return (
    <div className="eval-item">
      <span className="eval-label">{label}</span>
      <span className="eval-value" style={{ color }}>{value}</span>
      <span className="eval-desc">{desc}</span>
    </div>
  );
}

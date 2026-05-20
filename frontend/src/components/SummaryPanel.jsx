import { useEffect, useState } from "react";

export default function SummaryPanel() {
  const [summary, setSummary] = useState(null);

  useEffect(() => {
    fetch("http://localhost:8000/stats/summary")
      .then(r => r.json())
      .then(d => Object.keys(d).length && setSummary(d))
      .catch(() => {});
  }, []);

  if (!summary) return null;

  const dist = summary.category_distribution ?? {};

  return (
    <div className="card summary-card">
      <h3>🗂️ Session Summary</h3>
      <div className="summary-grid">
        <SumItem label="Total Records" value={summary.total_records} />
        <SumItem label="Avg Count" value={summary.avg_count} />
        <SumItem label="Peak Count" value={summary.max_count} />
        <SumItem label="Alerts" value={summary.alerts_triggered} highlight={summary.alerts_triggered > 0} />
      </div>
      {Object.keys(dist).length > 0 && (
        <div className="dist-row">
          {Object.entries(dist).map(([cat, cnt]) => (
            <span key={cat} className={`dist-chip dist-${cat.toLowerCase()}`}>
              {cat}: {cnt}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function SumItem({ label, value, highlight }) {
  return (
    <div className="sum-item">
      <span className="stat-label">{label}</span>
      <span className="stat-value" style={highlight ? { color: "#ef4444" } : {}}>{value}</span>
    </div>
  );
}

import { useEffect, useState } from "react";

export default function AlertHistory() {
  const [alerts, setAlerts] = useState([]);

  useEffect(() => {
    fetch("http://localhost:8000/alerts/history?limit=50")
      .then(r => r.json())
      .then(setAlerts)
      .catch(() => {});
  }, []);

  if (!alerts.length) return null;

  return (
    <div className="card alert-history-card">
      <h3>🚨 Alert History <span className="badge badge-red">{alerts.length}</span></h3>
      <div className="table-wrap">
        <table className="history-table">
          <thead>
            <tr>
              <th>Time</th>
              <th>Frame</th>
              <th>Count</th>
              <th>Level</th>
              <th>Message</th>
            </tr>
          </thead>
          <tbody>
            {alerts.map((a, i) => (
              <tr key={i} className="alert-row">
                <td>{a.timestamp.slice(11, 19)}</td>
                <td>{a.frame_id}</td>
                <td>{Math.round(a.person_count)}</td>
                <td style={{ color: "#ef4444" }}>{a.density_category}</td>
                <td style={{ color: "#fca5a5", fontSize: "0.78rem" }}>{a.alert}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

const CAT_COLOR = { Low: "#22c55e", Medium: "#f59e0b", High: "#ef4444" };

export default function HistoryTable({ rows }) {
  if (!rows.length) return null;

  return (
    <div className="card history-card">
      <h3>📋 Recent Records</h3>
      <div className="table-wrap">
        <table className="history-table">
          <thead>
            <tr>
              <th>Time</th>
              <th>Frame</th>
              <th>Count</th>
              <th>Level</th>
              <th>Tracks</th>
              <th>Alert</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i}>
                <td>{r.timestamp.slice(11, 19)}</td>
                <td>{r.frame_id}</td>
                <td>{Math.round(r.person_count)}</td>
                <td style={{ color: CAT_COLOR[r.density_category] }}>{r.density_category}</td>
                <td>{r.active_tracks}</td>
                <td>{r.alert ? "⚠️" : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

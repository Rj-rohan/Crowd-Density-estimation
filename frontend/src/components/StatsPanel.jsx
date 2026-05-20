const CATEGORY_COLOR = { Low: "#22c55e", Medium: "#f59e0b", High: "#ef4444" };

export default function StatsPanel({ stats }) {
  if (!stats) return (
    <div className="card stats-card">
      <h3>📊 Statistics</h3>
      <p className="muted">Waiting for data…</p>
    </div>
  );

  const color = CATEGORY_COLOR[stats.density_category] ?? "#888";

  return (
    <div className="card stats-card">
      <h3>📊 Statistics</h3>
      <div className="stat-grid">
        <StatItem label="People Count" value={Math.round(stats.person_count)} icon="👥" />
        <StatItem label="Density Score" value={stats.density_score.toFixed(1)} icon="📈" />
        <StatItem label="Active Tracks" value={stats.active_tracks} icon="🔍" />
        <div className="stat-item">
          <span className="stat-icon">🚦</span>
          <span className="stat-label">Density Level</span>
          <span className="stat-value" style={{ color }}>{stats.density_category}</span>
        </div>
      </div>
    </div>
  );
}

function StatItem({ label, value, icon }) {
  return (
    <div className="stat-item">
      <span className="stat-icon">{icon}</span>
      <span className="stat-label">{label}</span>
      <span className="stat-value">{value}</span>
    </div>
  );
}

export default function TrackList({ trackIds }) {
  return (
    <div className="card track-card">
      <h3>🔖 Active Track IDs <span className="badge badge-blue">{trackIds.length}</span></h3>
      <div className="track-list">
        {trackIds.length === 0
          ? <span className="muted">No active tracks</span>
          : trackIds.map(id => <span key={id} className="track-chip">#{id}</span>)
        }
      </div>
    </div>
  );
}

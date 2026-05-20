export default function VideoPanel({
  frame, showHeatmap, onToggleHeatmap,
  onStartStream, onStopStream, onUpload, connected, uploading,
  videoRef, canvasRef
}) {
  return (
    <div className="card video-card">
      <div className="card-header">
        <span>📹 Live Feed</span>
        <div className="controls">
          <button onClick={onToggleHeatmap} className="btn btn-secondary">
            {showHeatmap ? "Show Tracked" : "Show Heatmap"}
          </button>
          {!connected
            ? <button onClick={onStartStream} className="btn btn-primary">Start Camera</button>
            : <button onClick={onStopStream} className="btn btn-danger">Stop</button>
          }
          <label className={`btn btn-secondary ${uploading ? "btn-disabled" : ""}`}>
            {uploading ? "Analyzing…" : "Upload Video"}
            <input type="file" accept="video/*" hidden disabled={uploading}
              onChange={e => e.target.files[0] && onUpload(e.target.files[0])} />
          </label>
        </div>
      </div>

      <div className="video-container">
        {frame
          ? <img src={frame} alt="feed" className="video-frame" />
          : <div className="video-placeholder">No feed — start camera or upload a video</div>
        }
      </div>

      <video ref={videoRef} autoPlay muted style={{ display: "none" }} />
      <canvas ref={canvasRef} style={{ display: "none" }} />
    </div>
  );
}

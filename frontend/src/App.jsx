import { useState, useEffect, useRef, useCallback } from "react";
import VideoPanel from "./components/VideoPanel";
import StatsPanel from "./components/StatsPanel";
import DensityGraph from "./components/DensityGraph";
import AlertBanner from "./components/AlertBanner";
import TrackList from "./components/TrackList";
import SummaryPanel from "./components/SummaryPanel";
import HistoryTable from "./components/HistoryTable";
import "./App.css";

const WS_URL = "ws://localhost:8000/ws/stream";
const API = "http://localhost:8000";

export default function App() {
  const [frame, setFrame] = useState(null);
  const [heatmap, setHeatmap] = useState(null);
  const [stats, setStats] = useState(null);
  const [history, setHistory] = useState([]);       // graph data
  const [dbRows, setDbRows] = useState([]);          // table rows from DB
  const [alert, setAlert] = useState(null);
  const [connected, setConnected] = useState(false);
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [uploading, setUploading] = useState(false);

  const wsRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const intervalRef = useRef(null);

  // Load existing DB records on mount
  useEffect(() => {
    fetch(`${API}/stats/recent?limit=50`)
      .then(r => r.json())
      .then(rows => {
        setDbRows(rows);
        // seed graph with stored data (reversed — DB returns newest first)
        const graphData = [...rows].reverse().map(r => ({ frame_id: r.frame_id, count: r.person_count }));
        setHistory(graphData.slice(-60));
      })
      .catch(() => {});
  }, []);

  const connectWS = useCallback(() => {
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;
    ws.onopen = () => setConnected(true);
    ws.onclose = () => { setConnected(false); stopStream(); };
    ws.onerror = () => ws.close();
    ws.onmessage = (e) => {
      const data = JSON.parse(e.data);
      setFrame(`data:image/jpeg;base64,${data.frame}`);
      setHeatmap(`data:image/jpeg;base64,${data.heatmap}`);
      setAlert(data.alert);
      setStats({
        person_count: data.person_count,
        density_score: data.density_score,
        density_category: data.density_category,
        active_tracks: data.active_tracks,
        track_ids: data.track_ids,
      });
      setHistory(prev => [...prev, { frame_id: data.frame_id, count: data.person_count }].slice(-60));
      setDbRows(prev => [{
        timestamp: new Date().toISOString(),
        frame_id: data.frame_id,
        person_count: data.person_count,
        density_category: data.density_category,
        active_tracks: data.active_tracks,
        alert: data.alert,
      }, ...prev].slice(0, 50));
    };
  }, []);

  const startStream = useCallback(async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    streamRef.current = stream;
    videoRef.current.srcObject = stream;
    connectWS();
    intervalRef.current = setInterval(() => {
      const canvas = canvasRef.current;
      const video = videoRef.current;
      if (!canvas || !video || wsRef.current?.readyState !== 1) return;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext("2d").drawImage(video, 0, 0);
      canvas.toBlob(blob => {
        blob?.arrayBuffer().then(buf => wsRef.current?.readyState === 1 && wsRef.current.send(buf));
      }, "image/jpeg", 0.7);
    }, 200);
  }, [connectWS]);

  const stopStream = useCallback(() => {
    clearInterval(intervalRef.current);
    streamRef.current?.getTracks().forEach(t => t.stop());
    wsRef.current?.close();
    setConnected(false);
  }, []);

  const handleUpload = useCallback(async (file) => {
    setUploading(true);
    setAlert(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch(`${API}/analyze/upload`, { method: "POST", body: form });
      const data = await res.json();
      if (data.results?.length) {
        const last = data.results[data.results.length - 1];
        setFrame(`data:image/jpeg;base64,${last.frame}`);
        setHeatmap(`data:image/jpeg;base64,${last.heatmap}`);
        setAlert(last.alert);
        setStats({
          person_count: last.person_count,
          density_score: last.density_score,
          density_category: last.density_category,
          active_tracks: last.active_tracks,
          track_ids: last.track_ids,
        });
        setHistory(data.results.map(r => ({ frame_id: r.frame_id, count: r.person_count })));
        setDbRows(data.results.map(r => ({
          timestamp: new Date().toISOString(),
          frame_id: r.frame_id,
          person_count: r.person_count,
          density_category: r.density_category,
          active_tracks: r.active_tracks,
          alert: r.alert,
        })).reverse());
      }
    } finally {
      setUploading(false);
    }
  }, []);

  useEffect(() => () => stopStream(), [stopStream]);

  return (
    <div className="dashboard">
      <header className="header">
        <h1>🎯 Crowd Detection & Tracking System</h1>
        <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
          <span className={`badge ${connected ? "badge-green" : "badge-red"}`}>
            {connected ? "● LIVE" : "○ OFFLINE"}
          </span>
          {uploading && <span className="badge badge-yellow">⏳ Analyzing…</span>}
        </div>
      </header>

      {alert && <AlertBanner message={alert} />}

      <div className="main-grid">
        <VideoPanel
          frame={showHeatmap ? heatmap : frame}
          showHeatmap={showHeatmap}
          onToggleHeatmap={() => setShowHeatmap(v => !v)}
          onStartStream={startStream}
          onStopStream={stopStream}
          onUpload={handleUpload}
          connected={connected}
          uploading={uploading}
          videoRef={videoRef}
          canvasRef={canvasRef}
        />
        <div className="right-panel">
          <StatsPanel stats={stats} />
          <TrackList trackIds={stats?.track_ids ?? []} />
          <SummaryPanel />
        </div>
      </div>

      <DensityGraph data={history} />
      <HistoryTable rows={dbRows} />
    </div>
  );
}

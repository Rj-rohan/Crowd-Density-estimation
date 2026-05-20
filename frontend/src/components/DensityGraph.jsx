import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from "recharts";

const ALERT_THRESHOLD = 50;

export default function DensityGraph({ data }) {
  return (
    <div className="card graph-card">
      <h3>📉 Crowd Count Over Time</h3>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={data} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#333" />
          <XAxis dataKey="frame_id" tick={{ fontSize: 11, fill: "#aaa" }} label={{ value: "Frame", position: "insideBottom", fill: "#aaa", fontSize: 11 }} />
          <YAxis tick={{ fontSize: 11, fill: "#aaa" }} />
          <Tooltip contentStyle={{ background: "#1e1e2e", border: "1px solid #444", color: "#fff" }} />
          <ReferenceLine y={ALERT_THRESHOLD} stroke="#ef4444" strokeDasharray="4 4" label={{ value: "Alert", fill: "#ef4444", fontSize: 11 }} />
          <Line type="monotone" dataKey="count" stroke="#6366f1" strokeWidth={2} dot={false} name="People" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

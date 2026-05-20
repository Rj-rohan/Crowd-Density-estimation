export default function AlertBanner({ message }) {
  if (!message) return null;
  return (
    <div className="alert-banner">
      <span className="alert-icon">🚨</span>
      <span>{message}</span>
    </div>
  );
}

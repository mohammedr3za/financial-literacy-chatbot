export const config = { runtime: "edge" };
export default async function handler() {
  const backend = process.env.FINBOT_BACKEND_URL;
  let online = false, host = null, model = null;
  if (backend) {
    try {
      const r = await fetch(`${backend.replace(/\/$/, "")}/api/health`, { signal: AbortSignal.timeout(4000) });
      online = r.ok;
      if (r.ok) { try { const j = await r.json(); host = j.host || null; model = j.model || null; } catch (_) { /* older backend */ } }
    } catch (_) { online = false; }
  }
  return new Response(JSON.stringify({ online, host, model }), { headers: { "Content-Type": "application/json", "Cache-Control": "no-store" } });
}

export const config = { runtime: "edge" };
export default async function handler() {
  const backend = process.env.FINBOT_BACKEND_URL;
  let online = false;
  if (backend) {
    try {
      const r = await fetch(`${backend.replace(/\/$/, "")}/api/health`, { signal: AbortSignal.timeout(4000) });
      online = r.ok;
    } catch (_) { online = false; }
  }
  return new Response(JSON.stringify({ online }), { headers: { "Content-Type": "application/json", "Cache-Control": "no-store" } });
}

// Vercel Edge function: adds the secret token and forwards the event stream. The browser never sees the token or backend address.
export const config = { runtime: "edge" };
const json = (o, s) => new Response(JSON.stringify(o), { status: s, headers: { "Content-Type": "application/json" } });

export default async function handler(req) {
  if (req.method !== "POST") return json({ detail: "Method not allowed." }, 405);
  const backend = process.env.FINBOT_BACKEND_URL, token = process.env.FINBOT_API_TOKEN;
  if (!backend || !token) return json({ detail: "FinBot is not configured." }, 503);
  const body = await req.text();
  if (body.length > 400_000) return json({ detail: "Request too large." }, 413);
  const ip = (req.headers.get("x-forwarded-for") || "").split(",")[0].trim() || "unknown";
  let upstream;
  try {
    upstream = await fetch(`${backend.replace(/\/$/, "")}/api/chat`, {
      method: "POST", body,
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}`, "X-Client-IP": ip },
    });
  } catch (_) {
    return json({ detail: "FinBot is offline right now." }, 503);
  }
  return new Response(upstream.body, {
    status: upstream.status,
    headers: { "Content-Type": upstream.headers.get("content-type") || "application/json", "Cache-Control": "no-cache" },
  });
}

// Vercel Edge function: forwards a "Report a problem" request with the secret token.
export const config = { runtime: "edge" };
const json = (o, s) => new Response(JSON.stringify(o), { status: s, headers: { "Content-Type": "application/json" } });

export default async function handler(req) {
  if (req.method !== "POST") return json({ detail: "Method not allowed." }, 405);
  const backend = process.env.FINBOT_BACKEND_URL, token = process.env.FINBOT_API_TOKEN;
  if (!backend || !token) return json({ detail: "FinBot is not configured." }, 503);
  const body = await req.text();
  if (body.length > 5_000) return json({ detail: "Report too long." }, 413);
  const ip = (req.headers.get("x-forwarded-for") || "").split(",")[0].trim() || "unknown";
  try {
    const r = await fetch(`${backend.replace(/\/$/, "")}/api/report`, {
      method: "POST", body,
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}`, "X-Client-IP": ip },
    });
    return new Response(await r.text(), { status: r.status, headers: { "Content-Type": "application/json" } });
  } catch (_) {
    return json({ detail: "FinBot is offline right now." }, 503);
  }
}

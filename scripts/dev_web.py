"""Local stand-in for Vercel: serves web/ and proxies /api/* to the FinBot API with the secret token (same as web/api/*.js).

FINBOT_API_TOKEN=... uv run uvicorn scripts.dev_web:app --port 3000   (API must be running on FINBOT_BACKEND_URL)
"""
import os

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

BACKEND = os.environ.get("FINBOT_BACKEND_URL", "http://127.0.0.1:8080").rstrip("/")
TOKEN = os.environ["FINBOT_API_TOKEN"]
app = FastAPI()


@app.get("/api/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=4) as c:
            return {"online": (await c.get(f"{BACKEND}/api/health")).status_code == 200}
    except httpx.HTTPError:
        return {"online": False}


@app.post("/api/chat")
async def chat(req: Request):
    client = httpx.AsyncClient(timeout=None)
    upstream = await client.send(client.build_request(
        "POST", f"{BACKEND}/api/chat", content=await req.body(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {TOKEN}", "X-Client-IP": req.client.host}), stream=True)

    async def relay():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()
    return StreamingResponse(relay(), status_code=upstream.status_code, media_type=upstream.headers.get("content-type"))


app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "..", "web"), html=True), name="web")

"""Compare candidate local models on the real pipeline (SPEC section 2 fallbacks).

uv run python scripts/compare_models.py qwen3.5:9b-q4_K_M qwen3:8b qwen3.5:4b
For each model: the 6 acceptance prompts through FinBot (with document search), plus writing speed and memory.
"""
import json
import os
import statistics
import sys
import time
import urllib.request
import uuid

OLLAMA = "http://localhost:11434"


def ollama(path, payload=None):
    req = urllib.request.Request(OLLAMA + path, data=json.dumps(payload).encode() if payload else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read() or b"{}")


def unload_all():
    for m in ollama("/api/ps").get("models", []):
        if m["name"] != "bge-m3:latest":
            ollama("/api/generate", {"model": m["name"], "keep_alive": 0})
    time.sleep(2)


def main(models):
    from finbot.cli import ACCEPTANCE, check
    rows = []
    for model in models:
        unload_all()
        os.environ["FINBOT_MODEL"] = model
        from finbot.graph import FinBot
        bot = FinBot()
        bot.ask("hi", thread_id="warmup")  # load the model so the first timed answer is fair
        passed, lat, details = 0, [], []
        for key, q in ACCEPTANCE:
            t = bot.ask(q, thread_id=f"cmp-{key}-{uuid.uuid4().hex[:6]}")
            ok, why = check(key, t)
            passed += ok
            lat.append(t["latency_s"])
            details.append(f"    {'PASS' if ok else 'FAIL'} {t['latency_s']:5.1f}s rev={t['revisions']} {key}: {why[:110]}")
        payload = {"model": model, "prompt": "Explain compound interest in 100 words.", "stream": False,
                   "options": {"num_ctx": 16384}, "think": False}
        try:
            r = ollama("/api/generate", payload)
        except Exception:  # models without a thinking switch reject the "think" field
            payload.pop("think")
            r = ollama("/api/generate", payload)
        tps = r["eval_count"] / (r["eval_duration"] / 1e9)
        vram = next((m.get("size_vram", 0) for m in ollama("/api/ps")["models"] if m["name"].startswith(model.split(":")[0])), 0)
        rows.append((model, passed, statistics.median(lat), max(lat), tps, vram / 2**30))
        print(f"\n{model}: {passed}/6 acceptance, median {statistics.median(lat):.1f}s, max {max(lat):.1f}s, "
              f"{tps:.1f} tok/s writing, {vram / 2**30:.1f} GB", flush=True)
        print("\n".join(details), flush=True)
    print("\n| Model | Acceptance | Median answer | Slowest | Writing speed | Memory |\n|---|---|---|---|---|---|")
    for m, p, med, mx, tps, gb in rows:
        print(f"| {m} | {p}/6 | {med:.1f} s | {mx:.1f} s | {tps:.1f} tok/s | {gb:.1f} GB |")


if __name__ == "__main__":
    main(sys.argv[1:])

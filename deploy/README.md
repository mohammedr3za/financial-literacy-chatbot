# Deploying FinBot on a Lambda GPU

FinBot runs in two places with the same code: the Mac (Streamlit app, `qwen3:8b`, offline) and a Lambda A100 (Qwen3.6-27B, public API for the Vercel site). See SPEC sections 2 and 7.2.

## Exhibition-day relaunch
1. Launch an A100 (40 GB) in `asia-south-1` with the `finbot-reza-mac` SSH key. It costs $1.99/h, so **terminate it afterwards**.
2. Load your SSH key once on the Mac with `ssh-add --apple-use-keychain ~/.ssh/id_ed25519`.
3. Run `deploy/gpu_model.sh <ip>` (about 10 minutes the first time) to install vLLM and start the model.
4. Run `deploy/gpu_backend.sh <ip>` to copy code and the public index, start the API, and open the HTTPS tunnel. It prints `TUNNEL_URL=…`.
5. Point the site at the backend and redeploy:
   ```bash
   cd web && printf '%s' "<TUNNEL_URL>" | npx vercel env add FINBOT_BACKEND_URL production --force && npx vercel deploy --prod --yes
   ```
6. Open https://finbot-blond.vercel.app on a phone and ask the EMI example.

The quick tunnel URL changes every time the tunnel restarts, so repeat step 5 after any restart.

## Before terminating the GPU instance
Download the conversation log too (SPEC 7.4); it is training data, so keep it out of git:
```bash
mkdir -p results/chatlog-gpu && rsync -az ubuntu@<ip>:~/finbot/results/chatlog/ results/chatlog-gpu/
```
Download the "Report a problem" submissions first; they live on the instance and are lost when it is terminated:
```bash
mkdir -p results/reports && scp ubuntu@<ip>:~/finbot/results/reports/reports.jsonl results/reports/gpu-$(date +%Y%m%d-%H%M).jsonl
uv run python scripts/show_reports.py --path results/reports/gpu-<timestamp>.jsonl
```
